import numpy as np
from numba import jit
from collections import namedtuple

## B1_v8_v10 (Pairs Momentum) trade kernel outputs. SINGLE long-only stream on the
## leg1/leg2 ratio (long ratio = +leg1, -leg2, dollar-neutral): res (1/0), per-leg trade
## prices, per-bar trade cost, trade allocation (tal, carries between bars; NOT reset on
## exit — cost scaling uses it).
##
## There is no second entry: B1 opens once, exits once, and holds a constant size for the
## life of the trade (verified on the frozen goldens — 1,169 non-zero position runs, zero
## of which change size while open). The reversal2 / QR31 dual-stream contract
## (res2 / traded_price3,4 / profit_target1 / order_tag2) does NOT apply here.
TradeOutputPairMomentumArray = namedtuple("TradeOutputArray", [
    "res_arr",
    "tradeprice1_arr",
    "tradeprice2_arr",
    "tcost_arr",
    "trade_allocation_arr",
])

TradeOutputPairMomentumLastOutput = namedtuple("TradeOutputLastOutput", [
    "res_lo",
    "tradeprice1_lo",
    "tradeprice2_lo",
    "tcost_lo",
    "trade_allocation_lo",
    # carried kernel state — everything the per-bar _ll kernel needs to resume from the
    # batch warm-up. The kernel reads no [i-1] features, so this is pure trade state;
    # note bars_in_trade / can_take_new_trade / developed are NOT recoverable from the
    # output arrays, which is the whole reason this namedtuple exists:
    "long_signal_on",        # long_on
    "can_take_new_trade",    # can_new
    "entry_median_lo",       # em  (entry median — hard-stop pivot AND the pan/peak base)
    "entry_atr_lo",          # ea  (entry atr_eq)
    "peak_price_lo",         # ppx (peak ratio candle close since entry)
    "entry_long_vol_lo",     # elv (entry long_vol; 1e9 sentinel when unset)
    "bars_in_trade_lo",      # bars
    "developed_lo",          # pmv (pan >= TT reached — latches time-exit -> trail-exit)
    # live-layer conveniences (DB price reporting):
    "same_close1_lo",
    "same_close2_lo",
])


class TradeOutputPairMomentumLastOutputCls:
    """Mutable mirror of TradeOutputPairMomentumLastOutput, held on the strategy object
    as `numba_cls` and splatted back into the per-bar kernel each minute. The defaults
    are the array kernel's own initial values, so a cold start and a warm resume agree."""

    def __init__(self, res_lo, tradeprice1_lo, tradeprice2_lo, tcost_lo, trade_allocation_lo,
                 long_signal_on=False, can_take_new_trade=True,
                 entry_median_lo=0.0, entry_atr_lo=0.0, peak_price_lo=0.0,
                 entry_long_vol_lo=1e9, bars_in_trade_lo=0, developed_lo=0,
                 same_close1_lo=0.0, same_close2_lo=0.0):
        self.res_lo = res_lo
        self.tradeprice1_lo = tradeprice1_lo
        self.tradeprice2_lo = tradeprice2_lo
        self.tcost_lo = tcost_lo
        self.trade_allocation_lo = trade_allocation_lo
        self.long_signal_on = long_signal_on
        self.can_take_new_trade = can_take_new_trade
        self.entry_median_lo = entry_median_lo
        self.entry_atr_lo = entry_atr_lo
        self.peak_price_lo = peak_price_lo
        self.entry_long_vol_lo = entry_long_vol_lo
        self.bars_in_trade_lo = bars_in_trade_lo
        self.developed_lo = developed_lo
        self.same_close1_lo = same_close1_lo
        self.same_close2_lo = same_close2_lo

    def __repr__(self) -> str:
        return (f"TradeOutputPairMomentumLastOutputCls(res_lo={self.res_lo}, "
                f"tradeprice1_lo={self.tradeprice1_lo}, tradeprice2_lo={self.tradeprice2_lo}, "
                f"tcost_lo={self.tcost_lo}, trade_allocation_lo={self.trade_allocation_lo}, "
                f"long_signal_on={self.long_signal_on}, can_take_new_trade={self.can_take_new_trade}, "
                f"entry_median_lo={self.entry_median_lo}, entry_atr_lo={self.entry_atr_lo}, "
                f"peak_price_lo={self.peak_price_lo}, entry_long_vol_lo={self.entry_long_vol_lo}, "
                f"bars_in_trade_lo={self.bars_in_trade_lo}, developed_lo={self.developed_lo}, "
                f"same_close1_lo={self.same_close1_lo}, same_close2_lo={self.same_close2_lo})")


@jit(nopython=True)
def cryptopairs_b1v8v10_long_iact(next1, sc1, p1, mlow, next2, sc2, spc, med, upper,
                                  atr, lv, zmed, tc, slip, alloc,
                                  T1, TT, xdde, grace, k, x, zthr, ez):
    """B1_v8_v10 kernel: long-only Keltner-channel breakout on the leg1/leg2 price ratio,
    evaluated on the minutely frame with p1 marking tf-bar closes. Verbatim transcription
    of the frozen reference kern_v7
    (crypto_sims/PRODUCTION/engines/core/vendored_b1_dmp.py). Note vs B1_v8_v9: v8_v10 has
    NO daily dominance gate — the zM/zm arrays and the `dom` entry term are gone; nothing
    else differs.

    THE KERNEL IS THE SPECIFICATION. Where config comments or writeups disagree with the
    body below, the body wins. In particular:
      * TT is NOT a partial take — nothing halves the position. It latches `pmv`, which
        swaps the undeveloped time stop for the developed DDE trail.
      * TT and DDE are multiples of TRAIL VOL (lv), not counts of candles. Only T1 is
        measured in candles.
      * Entry compares the tf-candle CLOSE (spc) to the band, not the minute high.
      * The hard stop is anchored to the channel centre AT ENTRY (em), not the entry price.
      * tal is written ONLY on the entry bar and carries forward unchanged — deliberately
        not reset on exit, because the exit's cost scaling still needs it.

    Per minute: can_new re-arms whenever mlow <= med. In position, the z-confirmed hard
    stop can fire intrabar ((zmed < zthr) & (mlow < em - x*ea)). On a tf-bar close
    (p1 == 1): bars++, peak update, pan = ((spc-em)/em)/lv; pan >= TT -> developed;
    undeveloped -> time stop at T1 bars; developed -> DDE trail when
    (ppx-spc)/ppx >= xdde * min(lv, elv). Entry on p1 & can_new & spc > upper & zmed > ez:
    w = clip(k/zz, 0, 1), tal = alloc[i]*w, fills at next*(1 +/- slip), cost tc charged on
    entry AND exit.

    Returns (TradeOutputPairMomentumArray, TradeOutputPairMomentumLastOutput)."""
    n = next1.shape[0]
    res = np.zeros(n); tp1 = np.empty(n); tp2 = np.empty(n); tcost = np.zeros(n); tal = np.zeros(n)
    tp1[0] = sc1[0]; tp2[0] = sc2[0]
    long_on = False; can_new = True
    em = 0.0; ea = 0.0; ppx = 0.0; elv = 1e9; bars = 0; pmv = 0
    for i in range(1, n):
        tal[i] = tal[i - 1]
        if mlow[i] <= med[i]: can_new = True
        if long_on:
            if (zmed[i] < zthr) and (em > 0.0) and (mlow[i] < em - x * ea):
                long_on = False; res[i] = 0; tp1[i] = next1[i] * (1 - slip); tp2[i] = next2[i] * (1 + slip); tcost[i] = tc; continue
            if p1[i] == 1:
                bars += 1
                if spc[i] > ppx: ppx = spc[i]
                pan = ((spc[i] - em) / em) / lv[i] if (em > 0 and lv[i] > 0) else 0.0
                if pan >= TT: pmv = 1
                if pmv == 0:
                    if bars >= T1:
                        long_on = False; res[i] = 0; tp1[i] = next1[i] * (1 - slip); tp2[i] = next2[i] * (1 + slip); tcost[i] = tc; continue
                else:
                    trv = lv[i] if (lv[i] > 0.0 and lv[i] < elv) else elv   # trail_vol = min(current, entry)
                    if bars >= grace and ppx > 0.0 and (ppx - spc[i]) / ppx >= xdde * trv:
                        long_on = False; res[i] = 0; tp1[i] = next1[i] * (1 - slip); tp2[i] = next2[i] * (1 + slip); tcost[i] = tc; continue
            res[i] = 1; tp1[i] = sc1[i]; tp2[i] = sc2[i]
        else:
            if (p1[i] == 1 and can_new and (spc[i] > upper[i]) and (zmed[i] > ez)):
                em = med[i]; ea = atr[i]
                zz = (spc[i] - med[i]) / atr[i] if (med[i] > 0 and atr[i] > 0) else 0.0
                w = (k / zz) if zz > 0 else 1.0
                if w > 1.0: w = 1.0
                if w < 0.0: w = 0.0
                elv = lv[i] if lv[i] > 0 else 1e9                          # uncapped entry vol
                ppx = spc[i]; bars = 0; pmv = 0; can_new = False; long_on = True
                res[i] = 1; tp1[i] = next1[i] * (1 + slip); tp2[i] = next2[i] * (1 - slip); tcost[i] = tc; tal[i] = alloc[i] * w
            else:
                tp1[i] = sc1[i]; tp2[i] = sc2[i]
    ####
    trade_output_array = TradeOutputPairMomentumArray(res, tp1, tp2, tcost, tal)
    trade_output_last_output = TradeOutputPairMomentumLastOutput(
        res_lo=res[-1],
        tradeprice1_lo=tp1[-1],
        tradeprice2_lo=tp2[-1],
        tcost_lo=tcost[-1],
        trade_allocation_lo=tal[-1],
        long_signal_on=long_on,
        can_take_new_trade=can_new,
        entry_median_lo=em,
        entry_atr_lo=ea,
        peak_price_lo=ppx,
        entry_long_vol_lo=elv,
        bars_in_trade_lo=bars,
        developed_lo=pmv,
        same_close1_lo=sc1[-1],
        same_close2_lo=sc2[-1],
    )
    return trade_output_array, trade_output_last_output


@jit(nopython=True)
def cryptopairs_b1v8v10_long_iact_ll(next1, sc1, p1, mlow, next2, sc2, spc, med, upper,
                                     atr, lv, zmed, tc, slip, alloc,
                                     T1, TT, xdde, grace, k, x, zthr, ez,
                                     trade_allocation_1, long_signal_on, can_take_new_trade,
                                     entry_median, entry_atr, peak_price, entry_long_vol,
                                     bars_in_trade, developed):
    """Per-bar (last-line) form of cryptopairs_b1v8v10_long_iact — ONE minute per call.

    Inputs are the array kernel's 15 per-bar quantities as SCALARS, then its 8 per-cell
    scalars, then the 9 carried state fields (in TradeOutputPairMomentumLastOutput order)
    from the previous call. Returns a single TradeOutputPairMomentumLastOutput, which
    becomes the next call's state. Seeded from the array kernel's last_output at warm-up.

    ------------------------------------------------------------------------------
    READ THIS ABOUT `next1` / `next2` — THERE IS NO LOOKAHEAD HERE.
    ------------------------------------------------------------------------------
    The names are inherited from the array kernel, where they ARE forward-looking (the
    mean of the next 1-2 minutes' OHLC4 per leg). Running per-minute we obviously cannot
    see the future, so live.py passes the REALIZED fill price for THIS minute into these
    slots, out of its INPUT_DATA_TUPLE — the same substitution pair_lead_lag/live.py and
    sa_mft/live.py already make (`next_close1=coin1_hlc`, `asset_hlc3`). `update()`
    deliberately does not populate them; live.py owns the legs.

    That substitution is SAFE, and provably so rather than by convention: in the array
    kernel `next1`/`next2` occur in exactly two roles — `n = next1.shape[0]` (the array
    length) and `tp1 = next1*(1±slip)` / `tp2 = next2*(1∓slip)`. They appear inside ZERO
    conditions and are assigned into nothing but tp1/tp2. So they can only move the
    REPORTED trade price; they cannot change res, tal, long_on, can_new, or any entry or
    exit test. Batch and live therefore agree on every signal and differ only on the
    price attached to a fill.

    The names are kept identical to the array kernel so the two bodies stay diffable
    line by line.
    ------------------------------------------------------------------------------

    This is a line-for-line transliteration of the array kernel's loop body; the array
    kernel is the specification. The mapping (same pattern as
    cryptoasset_breakout_b1d_v14_last_line in utils/sa_directional_utils.py):
      * the loop's `tal[i] = tal[i-1]` carry   ->  `tal = trade_allocation_1` default
      * each `continue`                        ->  `done = True`
      * blocks that followed a `continue`      ->  guarded by `if (not done)`
      * trailing `res[i] = 1; tp1[i] = sc1[i]` ->  `if not done: res = 1.0; tp1 = sc1`
      * `res[i]` defaulting to 0 via np.zeros  ->  explicit `res = 0.0`

    Behaviours that MUST match the array kernel (each is a real failure mode):
      * tal is never reset on exit — written ONLY in the entry branch, because the exit's
        cost scaling still needs the size booked at entry.
      * can_new re-arms BEFORE the long/flat split, so it can flip mid-trade.
      * bars_in_trade++, the peak update and the `developed` latch all happen before the
        exit tests, and their mutated values are returned even when an exit fires — the
        array kernel's locals persist across `continue` in exactly the same way.
      * the hard stop skips the whole p1 block (the array kernel's `continue`).
      * entry_long_vol is the UNCAPPED entry vol, 1e9 sentinel when lv <= 0."""
    # Defaults for a "no change this bar" minute; the branches below overwrite them.
    res = 0.0
    tp1 = sc1
    tp2 = sc2
    tcost = 0.0
    tal = trade_allocation_1          # carries; NOT reset on exit
    done = False

    if mlow <= med:
        can_take_new_trade = True

    if long_signal_on:
        # z-confirmed hard stop — checked EVERY minute, anchored to the entry CENTRE.
        if (zmed < zthr) and (entry_median > 0.0) and (mlow < entry_median - x * entry_atr):
            long_signal_on = False
            res = 0.0
            tp1 = next1 * (1 - slip)      # exit fill: realized price this minute, live
            tp2 = next2 * (1 + slip)
            tcost = tc
            done = True

        # Candle-close work. Skipped entirely when the hard stop fired above.
        if (not done) and p1 == 1:
            bars_in_trade += 1
            if spc > peak_price:
                peak_price = spc
            pan = ((spc - entry_median) / entry_median) / lv if (entry_median > 0 and lv > 0) else 0.0
            if pan >= TT:
                developed = 1
            if developed == 0:
                if bars_in_trade >= T1:
                    long_signal_on = False
                    res = 0.0
                    tp1 = next1 * (1 - slip)
                    tp2 = next2 * (1 + slip)
                    tcost = tc
                    done = True
            else:
                trv = lv if (lv > 0.0 and lv < entry_long_vol) else entry_long_vol   # trail_vol = min(current, entry)
                if bars_in_trade >= grace and peak_price > 0.0 and (peak_price - spc) / peak_price >= xdde * trv:
                    long_signal_on = False
                    res = 0.0
                    tp1 = next1 * (1 - slip)
                    tp2 = next2 * (1 + slip)
                    tcost = tc
                    done = True

        if not done:
            # Holding: position stays on, marked at the minute closes.
            res = 1.0
            tp1 = sc1
            tp2 = sc2
    else:
        if (p1 == 1 and can_take_new_trade and (spc > upper) and (zmed > ez)):
            entry_median = med
            entry_atr = atr
            zz = (spc - med) / atr if (med > 0 and atr > 0) else 0.0
            w = (k / zz) if zz > 0 else 1.0
            if w > 1.0:
                w = 1.0
            if w < 0.0:
                w = 0.0
            entry_long_vol = lv if lv > 0 else 1e9                    # uncapped entry vol
            peak_price = spc
            bars_in_trade = 0
            developed = 0
            can_take_new_trade = False
            long_signal_on = True
            res = 1.0
            tp1 = next1 * (1 + slip)      # entry fill: realized price this minute, live
            tp2 = next2 * (1 - slip)
            tcost = tc
            tal = alloc * w
        else:
            tp1 = sc1
            tp2 = sc2
            res = 0.0

    return TradeOutputPairMomentumLastOutput(
        res_lo=res,
        tradeprice1_lo=tp1,
        tradeprice2_lo=tp2,
        tcost_lo=tcost,
        trade_allocation_lo=tal,
        long_signal_on=long_signal_on,
        can_take_new_trade=can_take_new_trade,
        entry_median_lo=entry_median,
        entry_atr_lo=entry_atr,
        peak_price_lo=peak_price,
        entry_long_vol_lo=entry_long_vol,
        bars_in_trade_lo=bars_in_trade,
        developed_lo=developed,
        same_close1_lo=sc1,
        same_close2_lo=sc2,
    )
