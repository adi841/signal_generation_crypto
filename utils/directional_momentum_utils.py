import numpy as np
from numba import jit
from collections import namedtuple

## DMP_v3_2 (Directional Momentum) trade kernel outputs. SINGLE ASSET, so there is ONE
## trade price, not a per-leg pair: res (+1/0 long stream, -1/0 short stream), trade price,
## per-bar trade cost, trade allocation (tal, carries between bars; NOT reset on exit —
## cost scaling uses it).
##
## TWO KERNELS, ONE SHAPE. A DMP cell runs a long Keltner breakout and its exact mirror
## short simultaneously, as two independent models with their own parent_stratid. They are
## transcribed here as two explicit kernels rather than one sign-parameterised kernel: the
## mirror is precisely where a transcription bug hides (which extreme, which band, which
## side of the z gate), and a shared body would make every such bug invisible to a
## line-by-line diff against the frozen reference. The cost is ~60 duplicated lines.
##
## There is no second entry and no partial take: DMP opens once, exits once, and holds a
## constant size for the life of the trade. `TT` latches the trail; it does NOT halve the
## position. The reversal2 / QR1 dual-stream contract (res2 / order_tag2 / profit targets)
## does NOT apply — DMP's "second stream" is the other SIDE, which is a separate model.
TradeOutputDirectionalMomentumArray = namedtuple("TradeOutputArray", [
    "res_arr",
    "tradeprice_arr",
    "tcost_arr",
    "trade_allocation_arr",
])

TradeOutputDirectionalMomentumLastOutput = namedtuple("TradeOutputLastOutput", [
    "res_lo",
    "tradeprice_lo",
    "tcost_lo",
    "trade_allocation_lo",
    # carried kernel state — everything the per-bar _ll kernel needs to resume from the
    # batch warm-up. The kernels read no [i-1] features, so this is pure trade state; note
    # bars_in_trade / can_take_new_trade / developed are NOT recoverable from the output
    # arrays, which is the whole reason this namedtuple exists:
    "signal_on",             # on
    "can_take_new_trade",    # can_new
    "entry_median_lo",       # entry_med (hard-stop pivot AND the adv/extreme base)
    "entry_atr_lo",          # entry_atr (entry atr_eq)
    "extreme_price_lo",      # peak (LONG) / trough (SHORT) candle close since entry
    "entry_long_vol_lo",     # entry_lv  (entry long_vol; 1e9 sentinel when unset)
    "bars_in_trade_lo",      # bars
    "developed_lo",          # moved (adv >= TT reached — swaps time exit for the trail)
    # live-layer convenience (DB price reporting):
    "same_close_lo",
])

## Cold-start initial for the SHORT kernel's running extreme. kern_short seeds
## `trough = 1e18` where kern_long seeds `peak = 0.0`; this is the ONLY initial-state value
## that differs between the two sides, and getting it wrong is silent (a trough of 0.0 can
## never be beaten by a positive price, so the trail would never arm).
SHORT_TROUGH_INIT = 1e18
LONG_PEAK_INIT = 0.0


class TradeOutputDirectionalMomentumLastOutputCls:
    """Mutable mirror of TradeOutputDirectionalMomentumLastOutput, held on the strategy
    object as `numba_cls` and splatted back into the per-bar kernel each minute.

    The defaults are the array kernels' own initial values, so a cold start and a warm
    resume agree — with ONE side-dependent exception: `extreme_price_lo` defaults to the
    LONG kernel's `peak = 0.0`. A cold SHORT start needs `SHORT_TROUGH_INIT`. Use
    `cold_start(side)` rather than the bare constructor when there is no warm-up to resume
    from; in the normal path the object is built from `**last_output._asdict()` and every
    field is supplied explicitly, so the defaults never apply.
    """

    def __init__(self, res_lo, tradeprice_lo, tcost_lo, trade_allocation_lo,
                 signal_on=False, can_take_new_trade=True,
                 entry_median_lo=0.0, entry_atr_lo=0.0, extreme_price_lo=LONG_PEAK_INIT,
                 entry_long_vol_lo=1e9, bars_in_trade_lo=0, developed_lo=0,
                 same_close_lo=0.0):
        self.res_lo = res_lo
        self.tradeprice_lo = tradeprice_lo
        self.tcost_lo = tcost_lo
        self.trade_allocation_lo = trade_allocation_lo
        self.signal_on = signal_on
        self.can_take_new_trade = can_take_new_trade
        self.entry_median_lo = entry_median_lo
        self.entry_atr_lo = entry_atr_lo
        self.extreme_price_lo = extreme_price_lo
        self.entry_long_vol_lo = entry_long_vol_lo
        self.bars_in_trade_lo = bars_in_trade_lo
        self.developed_lo = developed_lo
        self.same_close_lo = same_close_lo

    @classmethod
    def cold_start(cls, side, res_lo=0.0, tradeprice_lo=0.0, tcost_lo=0.0,
                   trade_allocation_lo=0.0):
        """Flat initial state matching the array kernel's own locals for `side`."""
        assert side in ("LONG", "SHORT"), f"unknown side {side!r}"
        return cls(res_lo, tradeprice_lo, tcost_lo, trade_allocation_lo,
                   extreme_price_lo=LONG_PEAK_INIT if side == "LONG" else SHORT_TROUGH_INIT)

    def __repr__(self) -> str:
        return (f"TradeOutputDirectionalMomentumLastOutputCls(res_lo={self.res_lo}, "
                f"tradeprice_lo={self.tradeprice_lo}, tcost_lo={self.tcost_lo}, "
                f"trade_allocation_lo={self.trade_allocation_lo}, signal_on={self.signal_on}, "
                f"can_take_new_trade={self.can_take_new_trade}, "
                f"entry_median_lo={self.entry_median_lo}, entry_atr_lo={self.entry_atr_lo}, "
                f"extreme_price_lo={self.extreme_price_lo}, "
                f"entry_long_vol_lo={self.entry_long_vol_lo}, "
                f"bars_in_trade_lo={self.bars_in_trade_lo}, developed_lo={self.developed_lo}, "
                f"same_close_lo={self.same_close_lo})")


@jit(nopython=True)
def cryptoasset_dmpv32_long_iact(next_close, same_close, p1, minutely_low, asset_close,
                                 median_line, upper_line, atr_eq, long_vol, z_median,
                                 txn_cost, slippage, allocation,
                                 time_exit_bars, tier_thresh, dd_eq_mult,
                                 k_atr, x_stop_atr, z_stop_thresh, entry_z_thresh):
    """DMP_v3_2 LONG kernel: Keltner-channel breakout on a SINGLE asset, evaluated on the
    minutely frame with p1 marking tf-bar closes. Verbatim transcription of the frozen
    reference `kern_long` (crypto_sims/PRODUCTION/engines/core/vendored_b1_dmp.py:137-177).

    THE KERNEL IS THE SPECIFICATION. Where config comments or writeups disagree with the
    body below, the body wins. In particular:
      * TT (tier_thresh) is NOT a partial take — nothing halves the position. It latches
        `moved`, which swaps the undeveloped time stop for the developed DDE trail.
      * TT and dd_eq_mult are multiples of TRAIL VOL (long_vol), not counts of candles.
        Only time_exit_bars is measured in candles.
      * Entry compares the tf-candle CLOSE (asset_close) to the band, not the minute high.
      * The hard stop is anchored to the channel centre AT ENTRY (entry_med), not the entry
        price, and it is checked EVERY minute against the minute LOW.
      * `dd_eq_mult` already carries the per-side M factor (M_LONG=4.0 x DDE_LAW[tf]); the
        kernel does not know about M.
      * There is NO grace parameter. B1's kern_v7 has one; kern_long/kern_short do not, so
        the trail is live from the first developed candle.
      * tal is written ONLY on the entry bar and carries forward unchanged — deliberately
        not reset on exit, because the exit's cost scaling still needs it.

    Per minute: can_new re-arms whenever minutely_low <= median_line. In position, the
    z-confirmed hard stop can fire intrabar ((z_median < Z) & (low < entry_med - X*entry_atr)).
    On a tf-bar close (p1 == 1): bars++, peak update, adv = ((ac-entry_med)/entry_med)/lv;
    adv >= TT -> developed; undeveloped -> time stop at T1 bars; developed -> DDE trail when
    (peak-ac)/peak >= dd_eq_mult * min(lv, entry_lv). Entry on
    p1 & can_new & ac > upper_line & z_median > EZ: w = clip(k/zz, 0, 1),
    tal = allocation[i]*w, fills at next_close*(1 +/- slip), cost charged on entry AND exit.

    Returns (TradeOutputDirectionalMomentumArray, TradeOutputDirectionalMomentumLastOutput)."""
    n = next_close.shape[0]
    res = np.zeros(n); tradeprice = np.empty(n); total_tc = np.zeros(n); trade_allocation = np.zeros(n)
    tradeprice[0] = same_close[0]
    on = False; can_new = True
    entry_med = 0.0; entry_atr = 0.0; peak = 0.0; entry_lv = 1e9; bars = 0; moved = 0
    for i in range(1, n):
        trade_allocation[i] = trade_allocation[i - 1]
        if minutely_low[i] <= median_line[i]:
            can_new = True
        if on:
            if (z_median[i] < z_stop_thresh) and (entry_med > 0.0) and (minutely_low[i] < entry_med - x_stop_atr * entry_atr):
                on = False; res[i] = 0; tradeprice[i] = next_close[i] * (1 - slippage); total_tc[i] = txn_cost; continue
            if p1[i] == 1:
                bars += 1
                if asset_close[i] > peak: peak = asset_close[i]
                adv = ((asset_close[i] - entry_med) / entry_med) / long_vol[i] if (entry_med > 0.0 and long_vol[i] > 0.0) else 0.0
                if adv >= tier_thresh: moved = 1
                if moved == 0:
                    if bars >= time_exit_bars:
                        on = False; res[i] = 0; tradeprice[i] = next_close[i] * (1 - slippage); total_tc[i] = txn_cost; continue
                else:
                    tv = long_vol[i] if (long_vol[i] > 0.0 and long_vol[i] < entry_lv) else entry_lv   # trail_vol = min(current, entry)
                    if peak > 0.0 and (peak - asset_close[i]) / peak >= dd_eq_mult * tv:
                        on = False; res[i] = 0; tradeprice[i] = next_close[i] * (1 - slippage); total_tc[i] = txn_cost; continue
            res[i] = 1; tradeprice[i] = same_close[i]
        else:
            if p1[i] == 1 and can_new and (asset_close[i] > upper_line[i]) and (z_median[i] > entry_z_thresh):
                zz = (asset_close[i] - median_line[i]) / atr_eq[i] if (median_line[i] > 0.0 and atr_eq[i] > 0.0) else 0.0
                w = (k_atr / zz) if zz > 0.0 else 1.0
                w = 1.0 if w > 1.0 else (0.0 if w < 0.0 else w)
                on = True; entry_med = median_line[i]; entry_atr = atr_eq[i]
                entry_lv = long_vol[i] if long_vol[i] > 0.0 else 1e9                          # uncapped entry vol
                peak = asset_close[i]; bars = 0; moved = 0; can_new = False
                tradeprice[i] = next_close[i] * (1 + slippage); res[i] = 1; total_tc[i] = txn_cost
                trade_allocation[i] = allocation[i] * w
            else:
                tradeprice[i] = same_close[i]
    ####
    trade_output_array = TradeOutputDirectionalMomentumArray(res, tradeprice, total_tc, trade_allocation)
    trade_output_last_output = TradeOutputDirectionalMomentumLastOutput(
        res_lo=res[-1],
        tradeprice_lo=tradeprice[-1],
        tcost_lo=total_tc[-1],
        trade_allocation_lo=trade_allocation[-1],
        signal_on=on,
        can_take_new_trade=can_new,
        entry_median_lo=entry_med,
        entry_atr_lo=entry_atr,
        extreme_price_lo=peak,
        entry_long_vol_lo=entry_lv,
        bars_in_trade_lo=bars,
        developed_lo=moved,
        same_close_lo=same_close[-1],
    )
    return trade_output_array, trade_output_last_output


@jit(nopython=True)
def cryptoasset_dmpv32_short_iact(next_close, same_close, p1, minutely_high, asset_close,
                                  median_line, lower_line, atr_eq, long_vol, z_median,
                                  txn_cost, slippage, allocation,
                                  time_exit_bars, tier_thresh, dd_eq_mult,
                                  k_atr, x_stop_atr, z_stop_thresh, entry_z_thresh):
    """DMP_v3_2 SHORT kernel: the exact mirror of the long kernel. Verbatim transcription of
    the frozen reference `kern_short` (vendored_b1_dmp.py:180-221).

    Every mirrored term, listed so the diff against kern_long is auditable:
      * re-arm      minutely_HIGH >= median_line        (long: low <= median_line)
      * entry gate  asset_close < LOWER_line
                    and z_median < -entry_z_thresh      (long: > +entry_z_thresh)
      * hard stop   z_median > -z_stop_thresh
                    and minutely_HIGH > entry_med + X*entry_atr   (long: low < med - X*atr)
      * extreme     TROUGH, seeded 1e18 and ratcheted DOWN (long: peak, seeded 0.0, up)
      * develop     adv = ((entry_med - ac)/entry_med)/lv          (long: (ac - entry_med))
      * trail       (ac - trough)/trough >= dd_eq_mult * tv        (long: (peak - ac)/peak)
      * sizing zz   (median_line - ac)/atr_eq                      (long: (ac - median_line))
      * fills       entry next*(1-slip), exit next*(1+slip)        (long: mirrored)
      * res         -1 while in position                           (long: +1)

    At the frozen EZ = 0.0 the two entry z-gates are exact complements; at Z = -0.3 both
    stops arm on |z| > 0.3 in the losing direction. `dd_eq_mult` carries M_SHORT = 1.5 (vs
    M_LONG = 4.0) — a much tighter trail, which is why the goldens show ~3x more short
    trades than long.

    NOTE the surviving asymmetries — these are NOT mirror bugs, they are in the reference:
      * `long_vol` is a two-sided EWM std of log-returns; it is NOT mirrored.
      * the `entry_med > 0.0` stop guard and the `trough > 0.0` trail guard are both
        `> 0.0`, not sign-flipped — they are price-positivity guards, not direction tests.
      * `k_atr` (K) is positive on both sides; the direction lives in zz's numerator.

    Returns (TradeOutputDirectionalMomentumArray, TradeOutputDirectionalMomentumLastOutput)."""
    n = next_close.shape[0]
    res = np.zeros(n); tradeprice = np.empty(n); total_tc = np.zeros(n); trade_allocation = np.zeros(n)
    tradeprice[0] = same_close[0]
    on = False; can_new = True
    entry_med = 0.0; entry_atr = 0.0; trough = 1e18; entry_lv = 1e9; bars = 0; moved = 0
    for i in range(1, n):
        trade_allocation[i] = trade_allocation[i - 1]
        if minutely_high[i] >= median_line[i]:
            can_new = True
        if on:
            if (z_median[i] > -z_stop_thresh) and (entry_med > 0.0) and (minutely_high[i] > entry_med + x_stop_atr * entry_atr):
                on = False; res[i] = 0; tradeprice[i] = next_close[i] * (1 + slippage); total_tc[i] = txn_cost; continue
            if p1[i] == 1:
                bars += 1
                if asset_close[i] < trough: trough = asset_close[i]
                adv = ((entry_med - asset_close[i]) / entry_med) / long_vol[i] if (entry_med > 0.0 and long_vol[i] > 0.0) else 0.0
                if adv >= tier_thresh: moved = 1
                if moved == 0:
                    if bars >= time_exit_bars:
                        on = False; res[i] = 0; tradeprice[i] = next_close[i] * (1 + slippage); total_tc[i] = txn_cost; continue
                else:
                    tv = long_vol[i] if (long_vol[i] > 0.0 and long_vol[i] < entry_lv) else entry_lv   # trail_vol = min(current, entry)
                    if trough > 0.0 and (asset_close[i] - trough) / trough >= dd_eq_mult * tv:
                        on = False; res[i] = 0; tradeprice[i] = next_close[i] * (1 + slippage); total_tc[i] = txn_cost; continue
            res[i] = -1; tradeprice[i] = same_close[i]
        else:
            if p1[i] == 1 and can_new and (asset_close[i] < lower_line[i]) and (z_median[i] < -entry_z_thresh):
                zz = (median_line[i] - asset_close[i]) / atr_eq[i] if (median_line[i] > 0.0 and atr_eq[i] > 0.0) else 0.0
                w = (k_atr / zz) if zz > 0.0 else 1.0
                w = 1.0 if w > 1.0 else (0.0 if w < 0.0 else w)
                on = True; entry_med = median_line[i]; entry_atr = atr_eq[i]
                entry_lv = long_vol[i] if long_vol[i] > 0.0 else 1e9                          # uncapped entry vol
                trough = asset_close[i]; bars = 0; moved = 0; can_new = False
                tradeprice[i] = next_close[i] * (1 - slippage); res[i] = -1; total_tc[i] = txn_cost
                trade_allocation[i] = allocation[i] * w
            else:
                tradeprice[i] = same_close[i]
    ####
    trade_output_array = TradeOutputDirectionalMomentumArray(res, tradeprice, total_tc, trade_allocation)
    trade_output_last_output = TradeOutputDirectionalMomentumLastOutput(
        res_lo=res[-1],
        tradeprice_lo=tradeprice[-1],
        tcost_lo=total_tc[-1],
        trade_allocation_lo=trade_allocation[-1],
        signal_on=on,
        can_take_new_trade=can_new,
        entry_median_lo=entry_med,
        entry_atr_lo=entry_atr,
        extreme_price_lo=trough,
        entry_long_vol_lo=entry_lv,
        bars_in_trade_lo=bars,
        developed_lo=moved,
        same_close_lo=same_close[-1],
    )
    return trade_output_array, trade_output_last_output


## ---------------------------------------------------------------------------------
## Per-bar (last-line) forms — ONE minute per call.
##
## Inputs are the array kernel's 13 per-bar quantities as SCALARS, then its 7 per-cell
## scalars, then the 9 carried state fields (in TradeOutputDirectionalMomentumLastOutput
## order) from the previous call. Each returns a single
## TradeOutputDirectionalMomentumLastOutput, which becomes the next call's state. Seeded
## from the array kernel's last_output at warm-up.
##
## ------------------------------------------------------------------------------
## READ THIS ABOUT `next_close` — THERE IS NO LOOKAHEAD HERE.
## ------------------------------------------------------------------------------
## The name is inherited from the array kernel, where it IS forward-looking (the mean of
## the next 1-2 minutes' OHLC4). Running per-minute we obviously cannot see the future, so
## live.py passes the REALIZED fill price for THIS minute into that slot, out of its
## INPUT_DATA_TUPLE — the same substitution the sibling sleeves already make. `update()`
## deliberately does not populate it; live.py owns the prices.
##
## That substitution is SAFE, and provably so rather than by convention: in both array
## kernels `next_close` occurs in exactly two roles — `n = next_close.shape[0]` (the array
## length) and `tradeprice = next_close*(1±slip)`. It appears inside ZERO conditions and is
## assigned into nothing but tradeprice. So it can only move the REPORTED trade price; it
## cannot change res, tal, on, can_new, or any entry or exit test. Batch and live therefore
## agree on every signal and differ only on the price attached to a fill.
##
## This is the property that makes DMP the cleanest of the four sleeves to match live.
## QR1_v4 does NOT have it — its kernel reads a fill price inside the scale-in trigger, so
## its live path can only be knife-edge-close. DMP's can be exact.
## ------------------------------------------------------------------------------
##
## Both bodies are line-for-line transliterations of their array kernels; the array kernels
## are the specification. The mapping:
##   * the loop's `trade_allocation[i] = trade_allocation[i-1]` carry -> `tal = trade_allocation_1`
##   * each `continue`                        ->  `done = True`
##   * blocks that followed a `continue`      ->  guarded by `if (not done)`
##   * trailing `res[i] = ±1; tradeprice[i] = same_close[i]` -> `if not done: ...`
##   * `res[i]` defaulting to 0 via np.zeros  ->  explicit `res = 0.0`
##
## Behaviours that MUST match the array kernels (each is a real failure mode):
##   * tal is never reset on exit — written ONLY in the entry branch, because the exit's
##     cost scaling still needs the size booked at entry.
##   * can_new re-arms BEFORE the on/flat split, so it can flip mid-trade.
##   * bars_in_trade++, the extreme update and the `developed` latch all happen before the
##     exit tests, and their mutated values are returned even when an exit fires — the
##     array kernel's locals persist across `continue` in exactly the same way.
##   * the hard stop skips the whole p1 block (the array kernel's `continue`).
##   * entry_long_vol is the UNCAPPED entry vol, 1e9 sentinel when long_vol <= 0.
## ---------------------------------------------------------------------------------


@jit(nopython=True)
def cryptoasset_dmpv32_long_iact_ll(next_close, same_close, p1, minutely_low, asset_close,
                                    median_line, upper_line, atr_eq, long_vol, z_median,
                                    txn_cost, slippage, allocation,
                                    time_exit_bars, tier_thresh, dd_eq_mult,
                                    k_atr, x_stop_atr, z_stop_thresh, entry_z_thresh,
                                    trade_allocation_1, signal_on, can_take_new_trade,
                                    entry_median, entry_atr, extreme_price, entry_long_vol,
                                    bars_in_trade, developed):
    """Per-bar form of cryptoasset_dmpv32_long_iact. `extreme_price` is the running PEAK."""
    # Defaults for a "no change this bar" minute; the branches below overwrite them.
    res = 0.0
    tradeprice = same_close
    tcost = 0.0
    tal = trade_allocation_1          # carries; NOT reset on exit
    done = False

    if minutely_low <= median_line:
        can_take_new_trade = True

    if signal_on:
        # z-confirmed hard stop — checked EVERY minute, anchored to the entry CENTRE.
        if (z_median < z_stop_thresh) and (entry_median > 0.0) and (minutely_low < entry_median - x_stop_atr * entry_atr):
            signal_on = False
            res = 0.0
            tradeprice = next_close * (1 - slippage)      # exit fill: realized price this minute, live
            tcost = txn_cost
            done = True

        # Candle-close work. Skipped entirely when the hard stop fired above.
        if (not done) and p1 == 1:
            bars_in_trade += 1
            if asset_close > extreme_price:
                extreme_price = asset_close
            adv = ((asset_close - entry_median) / entry_median) / long_vol if (entry_median > 0.0 and long_vol > 0.0) else 0.0
            if adv >= tier_thresh:
                developed = 1
            if developed == 0:
                if bars_in_trade >= time_exit_bars:
                    signal_on = False
                    res = 0.0
                    tradeprice = next_close * (1 - slippage)
                    tcost = txn_cost
                    done = True
            else:
                tv = long_vol if (long_vol > 0.0 and long_vol < entry_long_vol) else entry_long_vol   # trail_vol = min(current, entry)
                if extreme_price > 0.0 and (extreme_price - asset_close) / extreme_price >= dd_eq_mult * tv:
                    signal_on = False
                    res = 0.0
                    tradeprice = next_close * (1 - slippage)
                    tcost = txn_cost
                    done = True

        if not done:
            # Holding: position stays on, marked at the minute close.
            res = 1.0
            tradeprice = same_close
    else:
        if (p1 == 1 and can_take_new_trade and (asset_close > upper_line) and (z_median > entry_z_thresh)):
            zz = (asset_close - median_line) / atr_eq if (median_line > 0.0 and atr_eq > 0.0) else 0.0
            w = (k_atr / zz) if zz > 0.0 else 1.0
            w = 1.0 if w > 1.0 else (0.0 if w < 0.0 else w)
            signal_on = True
            entry_median = median_line
            entry_atr = atr_eq
            entry_long_vol = long_vol if long_vol > 0.0 else 1e9      # uncapped entry vol
            extreme_price = asset_close
            bars_in_trade = 0
            developed = 0
            can_take_new_trade = False
            res = 1.0
            tradeprice = next_close * (1 + slippage)      # entry fill: realized price this minute, live
            tcost = txn_cost
            tal = allocation * w
        else:
            tradeprice = same_close
            res = 0.0

    return TradeOutputDirectionalMomentumLastOutput(
        res_lo=res,
        tradeprice_lo=tradeprice,
        tcost_lo=tcost,
        trade_allocation_lo=tal,
        signal_on=signal_on,
        can_take_new_trade=can_take_new_trade,
        entry_median_lo=entry_median,
        entry_atr_lo=entry_atr,
        extreme_price_lo=extreme_price,
        entry_long_vol_lo=entry_long_vol,
        bars_in_trade_lo=bars_in_trade,
        developed_lo=developed,
        same_close_lo=same_close,
    )


@jit(nopython=True)
def cryptoasset_dmpv32_short_iact_ll(next_close, same_close, p1, minutely_high, asset_close,
                                     median_line, lower_line, atr_eq, long_vol, z_median,
                                     txn_cost, slippage, allocation,
                                     time_exit_bars, tier_thresh, dd_eq_mult,
                                     k_atr, x_stop_atr, z_stop_thresh, entry_z_thresh,
                                     trade_allocation_1, signal_on, can_take_new_trade,
                                     entry_median, entry_atr, extreme_price, entry_long_vol,
                                     bars_in_trade, developed):
    """Per-bar form of cryptoasset_dmpv32_short_iact. `extreme_price` is the running TROUGH
    (cold-start value SHORT_TROUGH_INIT = 1e18, not 0.0)."""
    # Defaults for a "no change this bar" minute; the branches below overwrite them.
    res = 0.0
    tradeprice = same_close
    tcost = 0.0
    tal = trade_allocation_1          # carries; NOT reset on exit
    done = False

    if minutely_high >= median_line:
        can_take_new_trade = True

    if signal_on:
        # z-confirmed hard stop — checked EVERY minute, anchored to the entry CENTRE.
        if (z_median > -z_stop_thresh) and (entry_median > 0.0) and (minutely_high > entry_median + x_stop_atr * entry_atr):
            signal_on = False
            res = 0.0
            tradeprice = next_close * (1 + slippage)      # exit fill: realized price this minute, live
            tcost = txn_cost
            done = True

        # Candle-close work. Skipped entirely when the hard stop fired above.
        if (not done) and p1 == 1:
            bars_in_trade += 1
            if asset_close < extreme_price:
                extreme_price = asset_close
            adv = ((entry_median - asset_close) / entry_median) / long_vol if (entry_median > 0.0 and long_vol > 0.0) else 0.0
            if adv >= tier_thresh:
                developed = 1
            if developed == 0:
                if bars_in_trade >= time_exit_bars:
                    signal_on = False
                    res = 0.0
                    tradeprice = next_close * (1 + slippage)
                    tcost = txn_cost
                    done = True
            else:
                tv = long_vol if (long_vol > 0.0 and long_vol < entry_long_vol) else entry_long_vol   # trail_vol = min(current, entry)
                if extreme_price > 0.0 and (asset_close - extreme_price) / extreme_price >= dd_eq_mult * tv:
                    signal_on = False
                    res = 0.0
                    tradeprice = next_close * (1 + slippage)
                    tcost = txn_cost
                    done = True

        if not done:
            # Holding: position stays on, marked at the minute close.
            res = -1.0
            tradeprice = same_close
    else:
        if (p1 == 1 and can_take_new_trade and (asset_close < lower_line) and (z_median < -entry_z_thresh)):
            zz = (median_line - asset_close) / atr_eq if (median_line > 0.0 and atr_eq > 0.0) else 0.0
            w = (k_atr / zz) if zz > 0.0 else 1.0
            w = 1.0 if w > 1.0 else (0.0 if w < 0.0 else w)
            signal_on = True
            entry_median = median_line
            entry_atr = atr_eq
            entry_long_vol = long_vol if long_vol > 0.0 else 1e9      # uncapped entry vol
            extreme_price = asset_close
            bars_in_trade = 0
            developed = 0
            can_take_new_trade = False
            res = -1.0
            tradeprice = next_close * (1 - slippage)      # entry fill: realized price this minute, live
            tcost = txn_cost
            tal = allocation * w
        else:
            tradeprice = same_close
            res = 0.0

    return TradeOutputDirectionalMomentumLastOutput(
        res_lo=res,
        tradeprice_lo=tradeprice,
        tcost_lo=tcost,
        trade_allocation_lo=tal,
        signal_on=signal_on,
        can_take_new_trade=can_take_new_trade,
        entry_median_lo=entry_median,
        entry_atr_lo=entry_atr,
        extreme_price_lo=extreme_price,
        entry_long_vol_lo=entry_long_vol,
        bars_in_trade_lo=bars_in_trade,
        developed_lo=developed,
        same_close_lo=same_close,
    )
