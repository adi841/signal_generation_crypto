import numpy as np
from numba import jit
from collections import namedtuple

## QR31_v2 (Pairs Lead-Lag) trade kernel outputs. LONG-only mean-reversion cycles on the
## leg1/leg2 ratio (long ratio = +leg1, -leg2, dollar-neutral): arm at the upper band,
## enter at the middle, exit at the (pivot-ratcheted) band or on a z/ATR or corr stop.
##
## ONE real tranche per trade: at the frozen flags (sec_mode=2) the "second entry" is
## VIRTUAL — it books no position and no cost, it only records the fill it WOULD have
## got and moves the pivot to the mean of the two, dragging the stop and profit-target
## base down. res2 is therefore identically 0 in production; it is kept (with
## tradeprice3/4, tcost2) because the kernel is a verbatim transcription of the frozen
## reference and the sec_mode=0 path still populates them.
TradeOutputPairLeadLagArray = namedtuple("TradeOutputArray", [
    "res_arr",
    "res2_arr",
    "tradeprice1_arr",
    "tradeprice2_arr",
    "tradeprice3_arr",
    "tradeprice4_arr",
    "tcost1_arr",
    "tcost2_arr",
    "trade_allocation_arr",
    "exit_reason_arr",       # 0=none 1=z_atr stop 2=corr stop 3=band@signal 4=band@non-signal 5=target 6=clock
    "arm_state_arr",         # can_take_new_trade at END of bar (int8)
])

TradeOutputPairLeadLagLastOutput = namedtuple("TradeOutputLastOutput", [
    "res_lo",
    "res2_lo",
    "tradeprice1_lo",
    "tradeprice2_lo",
    "tradeprice3_lo",
    "tradeprice4_lo",
    "tcost1_lo",
    "tcost2_lo",
    "trade_allocation_lo",
    "exit_reason_lo",
    # carried kernel state — everything the per-bar _ll kernel needs to resume from the
    # batch warm-up. None of these are recoverable from the output arrays alone:
    "long_signal_on",         # in a trade
    "can_take_new_trade",     # armed (upper-band touch since last exit/stop, not consumed)
    "second_trade_taken",     # virtual second entry has fired for the open trade
    "pair1_traded_price_lo",  # first-entry fill RATIO (slippage-adjusted); pivot base
    "pair2_traded_price_lo",  # sec_mode=0 second-entry fill ratio (0.0 under sec_mode=2)
    "virtual_p2_lo",          # sec_mode=2 virtual second fill ratio (pivot averaging)
    "profit_target_lo",       # profit_target1 carry (tp at entry; 1000 => inert)
    "traded_price1_lo",       # per-leg fill-price carries (reporting only — the kernel
    "traded_price2_lo",       #  copies them forward on gate-blocked and in-trade bars;
    "traded_price3_lo",       #  they appear in no decision condition)
    "traded_price4_lo",
    # THE ALIASING CARRY. In the frozen kernel upper_line / arm_line / critical_line are
    # ONE array (the caller passes the same mutated copy twice and the kernel aliases
    # critical_line = upper_line). The in-trade ratchet `critical_line[i] = pivot` (when
    # the band sits below the pivot) therefore ALSO raises the arm test and the
    # band-exit re-arm test that read `[i-1]` on the NEXT bar. A per-bar twin can only
    # reproduce those [i-1] reads by carrying the last bar's post-mutation band value:
    "upper_eff_prev_lo",
    # bars-since counters (from the kernel's absolute entry_bar / last_arm_bar /
    # last_stop_bar). At the frozen flag values (sec_delay_min=0, arm_expiry_min=0,
    # cooldown_min=0, clock_minutes=0) every gate they feed is inert; carried anyway so
    # a resume is faithful to the reference state machine:
    "bars_since_entry_lo",
    "bars_since_arm_lo",
    "bars_since_stop_lo",
    # live-layer conveniences (DB price reporting):
    "same_close1_lo",
    "same_close2_lo",
])


class TradeOutputPairLeadLagLastOutputCls:
    """Mutable mirror of TradeOutputPairLeadLagLastOutput, held on the strategy object
    as `numba_cls` and splatted back into the per-bar kernel each minute. The defaults
    are the array kernel's own initial values (bars-since counters use large sentinels,
    mirroring last_stop_bar = -10**9; upper_eff_prev has no kernel initial — it is data —
    so it defaults NaN, which keeps every band comparison False until seeded). In
    production it is ALWAYS seeded from the batch warm-up."""

    def __init__(self, res_lo, res2_lo, tradeprice1_lo, tradeprice2_lo, tradeprice3_lo,
                 tradeprice4_lo, tcost1_lo, tcost2_lo, trade_allocation_lo, exit_reason_lo,
                 long_signal_on=False, can_take_new_trade=True, second_trade_taken=False,
                 pair1_traded_price_lo=0.0, pair2_traded_price_lo=0.0, virtual_p2_lo=0.0,
                 profit_target_lo=0.0,
                 traded_price1_lo=0.0, traded_price2_lo=0.0,
                 traded_price3_lo=0.0, traded_price4_lo=0.0,
                 upper_eff_prev_lo=float('nan'),
                 bars_since_entry_lo=10**9, bars_since_arm_lo=10**9, bars_since_stop_lo=10**9,
                 same_close1_lo=0.0, same_close2_lo=0.0):
        self.res_lo = res_lo
        self.res2_lo = res2_lo
        self.tradeprice1_lo = tradeprice1_lo
        self.tradeprice2_lo = tradeprice2_lo
        self.tradeprice3_lo = tradeprice3_lo
        self.tradeprice4_lo = tradeprice4_lo
        self.tcost1_lo = tcost1_lo
        self.tcost2_lo = tcost2_lo
        self.trade_allocation_lo = trade_allocation_lo
        self.exit_reason_lo = exit_reason_lo
        self.long_signal_on = long_signal_on
        self.can_take_new_trade = can_take_new_trade
        self.second_trade_taken = second_trade_taken
        self.pair1_traded_price_lo = pair1_traded_price_lo
        self.pair2_traded_price_lo = pair2_traded_price_lo
        self.virtual_p2_lo = virtual_p2_lo
        self.profit_target_lo = profit_target_lo
        self.traded_price1_lo = traded_price1_lo
        self.traded_price2_lo = traded_price2_lo
        self.traded_price3_lo = traded_price3_lo
        self.traded_price4_lo = traded_price4_lo
        self.upper_eff_prev_lo = upper_eff_prev_lo
        self.bars_since_entry_lo = bars_since_entry_lo
        self.bars_since_arm_lo = bars_since_arm_lo
        self.bars_since_stop_lo = bars_since_stop_lo
        self.same_close1_lo = same_close1_lo
        self.same_close2_lo = same_close2_lo

    def __repr__(self) -> str:
        return (f"TradeOutputPairLeadLagLastOutputCls(res_lo={self.res_lo}, "
                f"res2_lo={self.res2_lo}, tradeprice1_lo={self.tradeprice1_lo}, "
                f"tradeprice2_lo={self.tradeprice2_lo}, tradeprice3_lo={self.tradeprice3_lo}, "
                f"tradeprice4_lo={self.tradeprice4_lo}, tcost1_lo={self.tcost1_lo}, "
                f"tcost2_lo={self.tcost2_lo}, trade_allocation_lo={self.trade_allocation_lo}, "
                f"exit_reason_lo={self.exit_reason_lo}, long_signal_on={self.long_signal_on}, "
                f"can_take_new_trade={self.can_take_new_trade}, "
                f"second_trade_taken={self.second_trade_taken}, "
                f"pair1_traded_price_lo={self.pair1_traded_price_lo}, "
                f"pair2_traded_price_lo={self.pair2_traded_price_lo}, "
                f"virtual_p2_lo={self.virtual_p2_lo}, profit_target_lo={self.profit_target_lo}, "
                f"traded_price1_lo={self.traded_price1_lo}, traded_price2_lo={self.traded_price2_lo}, "
                f"traded_price3_lo={self.traded_price3_lo}, traded_price4_lo={self.traded_price4_lo}, "
                f"upper_eff_prev_lo={self.upper_eff_prev_lo}, "
                f"bars_since_entry_lo={self.bars_since_entry_lo}, "
                f"bars_since_arm_lo={self.bars_since_arm_lo}, "
                f"bars_since_stop_lo={self.bars_since_stop_lo}, "
                f"same_close1_lo={self.same_close1_lo}, same_close2_lo={self.same_close2_lo})")


@jit(nopython=True)
def qr31_z_atr_stop(z_ema, z_threshold, minutely_low, minutely_high, pivot_price,
                    x_eff, atr2, stop_mode, stop_pct):
    """The QR31 z-confirmed ATR stop, in ONE place.

    Shared by the array kernel and its per-bar `_ll` twin so the two cannot drift apart.
    The factor study measured this stop as the sleeve's largest contributor (-1.28 Sharpe,
    MaxDD -2.2% -> -4.5% when removed) even though its own realized PnL is deeply negative
    (-278 log units against +350 from the band exits) -- it realizes losses to prevent
    worse ones, which is why it must be replaced rather than simply deleted.

      stop_mode 0  PRODUCTION  (z_ema < z_thr) AND low  <= pivot*(1 - x_eff*atr2/100)
                1  INVERTED    (z_ema > z_thr) AND high >= pivot*(1 + x_eff*atr2/100)
                               -- stops out winners and holds losers
                2  PLAIN-ATR   low <= pivot*(1 - x_eff*atr2/100)   (z-confirmation dropped)
                3  FIXED-PCT   low <= pivot*(1 - stop_pct/100)     (vol-independent)
    """
    if stop_mode == 1:
        return (z_ema > z_threshold) and (minutely_high >= pivot_price * (1 + x_eff * atr2 / 100))
    if stop_mode == 2:
        return minutely_low <= pivot_price * (1 - x_eff * atr2 / 100)
    if stop_mode == 3:
        return minutely_low <= pivot_price * (1 - stop_pct / 100)
    return (z_ema < z_threshold) and (minutely_low <= pivot_price * (1 - x_eff * atr2 / 100))


@jit(nopython=True)
def cryptopairs_qr31v2_long_iact(
    next_close1, same_close1, tp, minutely_high, minutely_low, p1,
    next_close2, same_close2, txn_cost, m1, m2,
    upper_line, middle_line, atr, atr2,
    nbdevupdn, slippage, allocation,
    z_ema, z_threshold, x_atr, corr, corr_threshold,
    entry_c_threshold,
    arm_always, sec_off, clock_minutes, use_override, override_sig,
    sec_delay_min, sec_mode, arm_expiry_min,
    rearm_off, cooldown_min, arm_line,
    stop_mode=0, stop_pct=2.0, signal_invert=0,
):
    """QR31_v2 kernel: long-only band-cycle mean reversion on the leg1/leg2 price ratio,
    evaluated on the minutely frame with p1 marking tf-bar closes. VERBATIM transcription
    of the frozen reference v16_flex_c
    (crypto_sims/PRODUCTION/engines/core/kernel_qr31.py) — the loop body is
    statement-for-statement identical, including the dead `elif False` branch and the
    `critical_line = upper_line` alias. Only the return packaging differs (namedtuples
    instead of the reference's bare tuple of arrays). Verified AST-IDENTICAL to the
    frozen v16_flex_c: all 35 pre-loop statements and the entire for-loop (2,579 AST
    nodes) match exactly; bitwise parity confirmed on real BTC/AVAX cells at TF 15 and
    240 (11/11 output arrays, mutated band array, and LastOutput consistency).

    THE KERNEL IS THE SPECIFICATION. Operationally load-bearing facts:
      * CALLER CONTRACT: pass ONE array object (a copy of the upper band — the kernel
        MUTATES it) for BOTH `upper_line` and `arm_line`, exactly as the reference
        run_cell passes `upc` twice. With `critical_line = upper_line` inside, all
        three names are the SAME array; the in-trade ratchet `critical_line[i] = pivot`
        also raises the arm test and the band-exit re-arm test that read `[i-1]` on the
        next bar.
      * Frozen production flags: sec_mode=2 (virtual second entry), every other flag 0,
        entry_c_threshold=-1e18 (entry corr gate off), tp=1000 (profit target inert),
        m2=1 with m1=msig (+1 iff z_ema > 0 — the entry momentum gate).
      * A middle touch while ARMED but msig BAD consumes the arm (can_take_new_trade
        goes False without an entry).
      * After a band exit, re-arm is immediate ONLY if the exit bar's high also touched
        the (possibly ratcheted) upper band; stops always disarm.
      * The virtual second entry books NO fill and NO cost; res2 stays 0 under
        sec_mode=2. trade_allocation is written at the entry bar and carried while
        in-trade (including the exit bar, whose cost scaling needs it); on FLAT bars
        it is 0 — v16 has NO cross-trade tal carry (unlike B1's kern_v7).
      * next_close1/next_close2 appear in ZERO conditions directly — BUT they are NOT
        signal-inert (unlike B1): the entry books pair1_traded_price (and sec_mode=2
        books virtual_p2) from them WITH SLIPPAGE, and that pivot IS read by the z/ATR
        stop, the virtual-second trigger and the band ratchet. Fills — and the slippage
        they carry — therefore shift signal levels; the live layer pins the kernel's
        slippage to the FROZEN per-cell value for exactly this reason.
      * On candle-close minutes (p1 == 1) the kernel reads the PREVIOUS row's lines
        ([i-1]); otherwise the current projected row ([i]).

    exit_reason codes: 1=z_atr stop, 2=corr stop, 3=band hit @signal bar,
    4=band hit @non-signal bar, 5=profit target, 6=clock exit.

    Returns (TradeOutputPairLeadLagArray, TradeOutputPairLeadLagLastOutput)."""
    n = next_close1.shape[0]
    res  = np.zeros(n)
    res2 = np.zeros(n)
    tradeprice1 = np.empty(n); tradeprice2 = np.empty(n)
    tradeprice3 = np.empty(n); tradeprice4 = np.empty(n)
    total_tc1 = np.zeros(n); total_tc2 = np.zeros(n)
    trade_allocation = np.zeros(n)
    exit_reason = np.zeros(n, dtype=np.int32)
    arm_state = np.zeros(n, dtype=np.int8)

    signal_price1 = np.zeros(n); traded_price1 = np.zeros(n)
    signal_price2 = np.zeros(n); traded_price2 = np.zeros(n)
    signal_price3 = np.zeros(n); traded_price3 = np.zeros(n)
    signal_price4 = np.zeros(n); traded_price4 = np.zeros(n)
    pair1_traded_price = np.zeros(n)
    pair2_traded_price = np.zeros(n)
    profit_target1 = np.zeros(n)

    tradeprice1[0] = same_close1[0]; tradeprice2[0] = same_close2[0]
    tradeprice3[0] = same_close1[0]; tradeprice4[0] = same_close2[0]

    long_signal_on = False
    can_take_new_trade = True
    second_trade_taken = False
    entry_bar = -1
    last_arm_bar = 0
    virtual_p2 = 0.0
    last_stop_bar = -10**9

    critical_line = upper_line  # long-only path; alias preserved deliberately (matches production)

    for i in range(1, n):
        if p1[i] == 1:
            if minutely_high[i] >= arm_line[i-1]:
                can_take_new_trade = True
                last_arm_bar = i
        else:
            if minutely_high[i] >= arm_line[i]:
                can_take_new_trade = True
                last_arm_bar = i
        if arm_expiry_min > 0 and can_take_new_trade and (i - last_arm_bar) > arm_expiry_min:
            can_take_new_trade = False
        if arm_always == 1:
            can_take_new_trade = True

        # entry trigger condition (standard or override)
        if use_override == 1:
            entry_trig = (override_sig[i] == 1) and (long_signal_on == False)
            gate_ok = True
        else:
            entry_trig = (((minutely_low[i] <= middle_line[i-1] and p1[i] == 1) or
                           (minutely_low[i] <= middle_line[i] and p1[i] != 1))
                          and can_take_new_trade and long_signal_on == False
                          and corr[i] >= entry_c_threshold
                          and (i - last_stop_bar) >= cooldown_min)
            gate_ok = (m1[i] == 1 and m2 == 1)

        # ── First LONG entry ────────────────────────────────────────────────
        if entry_trig:
            if gate_ok:
                long_signal_on = True
                second_trade_taken = False
                entry_bar = i
                signal_price1[i] = same_close1[i]
                traded_price1[i] = next_close1[i] * (1 + slippage)
                signal_price2[i] = same_close2[i]
                traded_price2[i] = next_close2[i] * (1 - slippage)
                tradeprice1[i] = traded_price1[i]
                tradeprice2[i] = traded_price2[i]
                tradeprice3[i] = same_close1[i]
                tradeprice4[i] = same_close2[i]
                pair1_traded_price[i] = traded_price1[i] / traded_price2[i]
                pair2_traded_price[i] = 0.0
                res[i] = 1
                res2[i] = 0
                profit_target1[i] = tp[i]
                total_tc1[i] = txn_cost
                total_tc2[i] = 0.0
                trade_allocation[i] = allocation[i]
            else:
                traded_price1[i] = traded_price1[i-1]
                traded_price2[i] = traded_price2[i-1]
                tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
                tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]
                pair1_traded_price[i] = pair1_traded_price[i-1]
                pair2_traded_price[i] = 0.0
                can_take_new_trade = False
        # ── Skipped entries (midline but not armed) ─────────────────────────
        elif ((use_override == 0) and
              ((minutely_low[i] <= middle_line[i-1] and p1[i] == 1) or
               (minutely_low[i] <= middle_line[i] and p1[i] != 1)) and not long_signal_on):
            traded_price1[i] = traded_price1[i-1]
            traded_price2[i] = traded_price2[i-1]
            tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
            tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]
            pair1_traded_price[i] = pair1_traded_price[i-1]
            pair2_traded_price[i] = 0.0
        # ── IN LONG ─────────────────────────────────────────────────────────
        elif long_signal_on == True:
            signal_price1[i] = signal_price1[i-1]
            traded_price1[i] = traded_price1[i-1]
            signal_price2[i] = signal_price2[i-1]
            traded_price2[i] = traded_price2[i-1]
            pair1_traded_price[i] = pair1_traded_price[i-1]
            profit_target1[i] = profit_target1[i-1]
            trade_allocation[i] = trade_allocation[i-1]

            if (not second_trade_taken) and (sec_off == 0) and (i - entry_bar >= sec_delay_min):
                sec_trig = ((minutely_low[i] <= pair1_traded_price[i] * (1 - nbdevupdn * atr[i-1] / 100) and p1[i] == 1) or
                            (minutely_low[i] <= pair1_traded_price[i] * (1 - nbdevupdn * atr[i]   / 100) and p1[i] != 1))
                if sec_trig:
                    second_trade_taken = True
                    if sec_mode == 0:
                        res2[i] = 1
                        signal_price3[i] = same_close1[i]
                        traded_price3[i] = next_close1[i] * (1 + slippage)
                        signal_price4[i] = same_close2[i]
                        traded_price4[i] = next_close2[i] * (1 - slippage)
                        tradeprice3[i] = traded_price3[i]
                        tradeprice4[i] = traded_price4[i]
                        pair2_traded_price[i] = traded_price3[i] / traded_price4[i]
                        total_tc2[i] = txn_cost
                        avg_p = (pair1_traded_price[i] + pair2_traded_price[i]) / 2
                        profit_target1[i] = 100.0 * ((avg_p * (1 + tp[i] / 100.0)) / pair1_traded_price[i] - 1.0)
                    else:
                        if sec_mode == 2:
                            virtual_p2 = (next_close1[i] * (1 + slippage)) / (next_close2[i] * (1 - slippage))
                        res2[i] = 0
                        tradeprice3[i] = same_close1[i]
                        tradeprice4[i] = same_close2[i]
                        pair2_traded_price[i] = 0.0
                elif False:
                    pass
                else:
                    res2[i] = 0
                    tradeprice3[i] = same_close1[i]
                    tradeprice4[i] = same_close2[i]
                    pair2_traded_price[i] = 0.0
            else:
                if second_trade_taken and sec_mode == 0:
                    res2[i] = 1
                signal_price3[i] = signal_price3[i-1]
                traded_price3[i] = traded_price3[i-1]
                signal_price4[i] = signal_price4[i-1]
                traded_price4[i] = traded_price4[i-1]
                tradeprice3[i] = same_close1[i]
                tradeprice4[i] = same_close2[i]
                pair2_traded_price[i] = pair2_traded_price[i-1]

            if second_trade_taken and sec_mode == 0:
                pivot_price = (pair1_traded_price[i] + pair2_traded_price[i]) / 2.0
            elif second_trade_taken and sec_mode == 2:
                pivot_price = (pair1_traded_price[i] + virtual_p2) / 2.0
            else:
                pivot_price = pair1_traded_price[i]

            if critical_line[i] < pivot_price:
                critical_line[i] = pivot_price

            if clock_minutes > 0:
                # clock-exit mode: ONLY time-based exit
                if i - entry_bar >= clock_minutes:
                    res[i] = 0; res2[i] = 0
                    long_signal_on = False
                    tradeprice1[i] = next_close1[i] * (1 - slippage)
                    tradeprice2[i] = next_close2[i] * (1 + slippage)
                    if second_trade_taken and sec_mode == 0:
                        tradeprice3[i] = next_close1[i] * (1 - slippage)
                        tradeprice4[i] = next_close2[i] * (1 + slippage)
                        total_tc2[i] = txn_cost
                    else:
                        tradeprice3[i] = same_close1[i]
                        tradeprice4[i] = same_close2[i]
                    second_trade_taken = False
                    total_tc1[i] = txn_cost
                    exit_reason[i] = 6
                else:
                    res[i] = 1
                    if second_trade_taken:
                        res2[i] = 1
                    tradeprice1[i] = same_close1[i]
                    tradeprice2[i] = same_close2[i]
            else:
                x_eff = x_atr * 1.5 if (sec_mode == 1 and second_trade_taken) else x_atr
                z_atr_hit = qr31_z_atr_stop(z_ema[i], z_threshold, minutely_low[i],
                                            minutely_high[i], pivot_price, x_eff, atr2[i],
                                            stop_mode, stop_pct)
                corr_hit  = corr[i] < corr_threshold

                if z_atr_hit or corr_hit:
                    res[i]  = 0; res2[i] = 0
                    long_signal_on = False
                    tradeprice1[i] = next_close1[i] * (1 - slippage)
                    tradeprice2[i] = next_close2[i] * (1 + slippage)
                    if second_trade_taken and sec_mode == 0:
                        tradeprice3[i] = next_close1[i] * (1 - slippage)
                        tradeprice4[i] = next_close2[i] * (1 + slippage)
                        total_tc2[i] = txn_cost
                    else:
                        tradeprice3[i] = same_close1[i]
                        tradeprice4[i] = same_close2[i]
                    second_trade_taken = False
                    total_tc1[i] = txn_cost
                    can_take_new_trade = False
                    last_stop_bar = i
                    if z_atr_hit:
                        exit_reason[i] = 1
                    else:
                        exit_reason[i] = 2
                elif (minutely_high[i] >= critical_line[i-1] and p1[i] == 1) or \
                     (minutely_high[i] >= critical_line[i]   and p1[i] != 1) or \
                     (minutely_high[i] > pivot_price * (1 + profit_target1[i] / 100)):
                    res[i]  = 0; res2[i] = 0
                    long_signal_on = False
                    tradeprice1[i] = next_close1[i] * (1 - slippage)
                    tradeprice2[i] = next_close2[i] * (1 + slippage)
                    if second_trade_taken and sec_mode == 0:
                        tradeprice3[i] = next_close1[i] * (1 - slippage)
                        tradeprice4[i] = next_close2[i] * (1 + slippage)
                        total_tc2[i] = txn_cost
                    else:
                        tradeprice3[i] = same_close1[i]
                        tradeprice4[i] = same_close2[i]
                    second_trade_taken = False
                    total_tc1[i] = txn_cost
                    if rearm_off == 1:
                        can_take_new_trade = False
                    elif (minutely_high[i] >= upper_line[i-1] and p1[i] == 1) or \
                         (minutely_high[i] >= upper_line[i]   and p1[i] != 1):
                        can_take_new_trade = True
                    else:
                        can_take_new_trade = False
                    if (minutely_high[i] >= critical_line[i-1] and p1[i] == 1):
                        exit_reason[i] = 3
                    elif (minutely_high[i] >= critical_line[i]   and p1[i] != 1):
                        exit_reason[i] = 4
                    else:
                        exit_reason[i] = 5
                else:
                    res[i] = 1
                    if second_trade_taken:
                        res2[i] = 1
                    tradeprice1[i] = same_close1[i]
                    tradeprice2[i] = same_close2[i]
        # ── Idle ─────────────────────────────────────────────────────────────
        else:
            long_signal_on = False
            second_trade_taken = False
            tradeprice1[i] = same_close1[i]
            tradeprice2[i] = same_close2[i]
            tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]
            traded_price1[i] = traded_price1[i-1]
            traded_price2[i] = traded_price2[i-1]
            pair1_traded_price[i] = pair1_traded_price[i-1]
            pair2_traded_price[i] = 0.0

        arm_state[i] = 1 if can_take_new_trade else 0

    ####
    ## Direction inversion -- trade AGAINST this sleeve's own signal. Applied to the
    ## REPORTED position only, after the state machine has run. Both tranches flip, so the
    ## scale-in keeps its relationship to the first entry. Costs keep production sign.
    if signal_invert == 1:
        res = -res
        res2 = -res2
    trade_output_array = TradeOutputPairLeadLagArray(
        res, res2, tradeprice1, tradeprice2, tradeprice3, tradeprice4,
        total_tc1, total_tc2, trade_allocation, exit_reason, arm_state,
    )
    trade_output_last_output = TradeOutputPairLeadLagLastOutput(
        res_lo=res[n-1],
        res2_lo=res2[n-1],
        tradeprice1_lo=tradeprice1[n-1],
        tradeprice2_lo=tradeprice2[n-1],
        tradeprice3_lo=tradeprice3[n-1],
        tradeprice4_lo=tradeprice4[n-1],
        tcost1_lo=total_tc1[n-1],
        tcost2_lo=total_tc2[n-1],
        trade_allocation_lo=trade_allocation[n-1],
        exit_reason_lo=exit_reason[n-1],
        long_signal_on=long_signal_on,
        can_take_new_trade=can_take_new_trade,
        second_trade_taken=second_trade_taken,
        pair1_traded_price_lo=pair1_traded_price[n-1],
        pair2_traded_price_lo=pair2_traded_price[n-1],
        virtual_p2_lo=virtual_p2,
        profit_target_lo=profit_target1[n-1],
        traded_price1_lo=traded_price1[n-1],
        traded_price2_lo=traded_price2[n-1],
        traded_price3_lo=traded_price3[n-1],
        traded_price4_lo=traded_price4[n-1],
        upper_eff_prev_lo=upper_line[n-1],
        bars_since_entry_lo=(n-1) - entry_bar,
        bars_since_arm_lo=(n-1) - last_arm_bar,
        bars_since_stop_lo=(n-1) - last_stop_bar,
        same_close1_lo=same_close1[n-1],
        same_close2_lo=same_close2[n-1],
    )
    return trade_output_array, trade_output_last_output


@jit(nopython=True)
def cryptopairs_qr31v2_long_iact_ll(
    # per-bar scalars (the array kernel's per-bar quantities for THIS minute)
    next_close1, same_close1, tp, minutely_high, minutely_low, p1,
    next_close2, same_close2, txn_cost, m1, m2,
    upper_line, middle_line, atr, atr2,
    nbdevupdn, slippage, allocation,
    z_ema, z_threshold, x_atr, corr, corr_threshold,
    entry_c_threshold,
    arm_always, sec_off, clock_minutes, use_override, override_sig,
    sec_delay_min, sec_mode, arm_expiry_min,
    rearm_off, cooldown_min,
    # previous-MINUTE line values (the array kernel's [i-1] reads on p1 minutes).
    # upper_eff_prev is the POST-RATCHET band of the previous minute — in the array
    # kernel upper_line / arm_line / critical_line are ONE mutated array, so every
    # [i-1] band read (arm, exits, re-arm) sees the ratcheted value; there is no
    # separate arm_line argument here for exactly that reason.
    upper_eff_prev, middle_prev, atr_prev,
    # carried state (TradeOutputPairLeadLagLastOutput order)
    trade_allocation_1, long_signal_on, can_take_new_trade, second_trade_taken,
    pair1_traded_price_1, pair2_traded_price_1, virtual_p2, profit_target_1,
    traded_price1_1, traded_price2_1, traded_price3_1, traded_price4_1,
    bars_since_entry, bars_since_arm, bars_since_stop,
    stop_mode=0, stop_pct=2.0, signal_invert=0,
):
    """Per-bar (last-line) form of cryptopairs_qr31v2_long_iact — ONE minute per call.

    Inputs are the array kernel's per-bar quantities as SCALARS, then the frozen
    per-cell scalars / flags, then the previous-minute line values, then the carried
    state (in TradeOutputPairLeadLagLastOutput order) from the previous call. Returns a
    single TradeOutputPairLeadLagLastOutput, which becomes the next call's state.
    Seeded from the array kernel's last_output at warm-up.

    ------------------------------------------------------------------------------
    READ THIS ABOUT `next_close1` / `next_close2` — THERE IS NO LOOKAHEAD HERE.
    ------------------------------------------------------------------------------
    The names are inherited from the array kernel, where they ARE forward-looking (the
    mean of the next 1-2 minutes' OHLC4 per leg). Running per-minute we cannot see the
    future, so live.py passes the REALIZED fill price for THIS minute into these slots.
    Unlike B1, they are NOT reporting-only: the entry books
    `pair1_traded_price = next1*(1+slip)/(next2*(1-slip))` — the PIVOT that the z/ATR
    stop, the virtual-second trigger and the band ratchet all read. live.py therefore
    REFINES numba_cls.pair1_traded_price_lo (and virtual_p2_lo) over the T+1/T+2 fill
    window so the state converges to the batch benchmark two minutes after the event;
    they still appear in ZERO entry/exit CONDITIONS, so a divergence is possible only
    if a stop/trigger fires inside that 2-minute window at the estimate-vs-final gap.

    Translation of the array loop body (`continue` -> `done`; canonical references:
    cryptopairs_b1v8v10_long_iact_ll, cryptoasset_breakout_b1d_v14_last_line), plus the
    QR31-specific index facts, each mirrored exactly:
      * ALIASING: the arm test reads upper_eff_prev (p1) / RAW upper_line (non-p1 —
        this minute's ratchet has not run yet); then upper_eff = upper_line, and the
        in-trade ratchet raises upper_eff to the pivot; ALL later band reads (exits,
        re-arm) use upper_eff_prev (p1) / upper_eff (non-p1). upper_eff is returned as
        the next minute's upper_eff_prev.
      * [i-1] line reads on p1 minutes: entry middle test -> middle_prev; virtual
        trigger atr -> atr_prev. z_ema / corr / atr2 / allocation / tp are [i]-reads.
      * bars-since counters advance at call start and reset to 0 on their events; the
        gates they feed are all inert at the frozen flags (0s) but stay faithful.
      * tal is written at entry and carried while in-trade (incl. the exit bar); 0 on
        flat bars — v16 has NO cross-trade tal carry. pair2_traded_price is zeroed on
        every non-in-trade branch exactly as the array kernel does.
      * the array kernel's dead `elif False: pass` branch is preserved."""
    # counters advance one minute (array kernel: i grows by 1 vs the stored bar indices)
    bars_since_entry += 1
    bars_since_arm += 1
    bars_since_stop += 1

    # defaults; branches below overwrite exactly as the array kernel's writes do
    res = 0.0
    res2 = 0.0
    tp1 = same_close1
    tp2 = same_close2
    tp3 = same_close1
    tp4 = same_close2
    tcost1 = 0.0
    tcost2 = 0.0
    # v16 tal semantics (UNLIKE B1's kern_v7 universal carry): the array kernel
    # zero-initialises trade_allocation and writes it ONLY in the entry branch and at
    # the top of the in-long branch — so it carries within a trade (incl. the exit
    # bar, whose cost scaling needs it) and is 0 on every flat bar.
    tal = 0.0
    exit_reason = 0

    pair1_traded_price = pair1_traded_price_1
    pair2_traded_price = pair2_traded_price_1
    profit_target = profit_target_1
    traded_price1 = traded_price1_1
    traded_price2 = traded_price2_1
    traded_price3 = traded_price3_1
    traded_price4 = traded_price4_1

    # ── arm block (reads the PRE-ratchet band for [i], POST-ratchet for [i-1]) ──
    if p1 == 1:
        if minutely_high >= upper_eff_prev:
            can_take_new_trade = True
            bars_since_arm = 0
    else:
        if minutely_high >= upper_line:
            can_take_new_trade = True
            bars_since_arm = 0
    if arm_expiry_min > 0 and can_take_new_trade and bars_since_arm > arm_expiry_min:
        can_take_new_trade = False
    if arm_always == 1:
        can_take_new_trade = True

    # this minute's effective band; the in-trade ratchet below may raise it
    upper_eff = upper_line

    # entry trigger condition (standard or override)
    if use_override == 1:
        entry_trig = (override_sig == 1) and (long_signal_on == False)
        gate_ok = True
    else:
        entry_trig = (((minutely_low <= middle_prev and p1 == 1) or
                       (minutely_low <= middle_line and p1 != 1))
                      and can_take_new_trade and long_signal_on == False
                      and corr >= entry_c_threshold
                      and bars_since_stop >= cooldown_min)
        gate_ok = (m1 == 1 and m2 == 1)

    # ── First LONG entry ────────────────────────────────────────────────
    if entry_trig:
        if gate_ok:
            long_signal_on = True
            second_trade_taken = False
            bars_since_entry = 0
            traded_price1 = next_close1 * (1 + slippage)
            traded_price2 = next_close2 * (1 - slippage)
            tp1 = traded_price1
            tp2 = traded_price2
            tp3 = same_close1
            tp4 = same_close2
            pair1_traded_price = traded_price1 / traded_price2
            pair2_traded_price = 0.0
            res = 1.0
            res2 = 0.0
            profit_target = tp
            tcost1 = txn_cost
            tcost2 = 0.0
            tal = allocation
        else:
            # gate-blocked: reporting carries, the arm is CONSUMED
            tp1 = same_close1
            tp2 = same_close2
            tp3 = same_close1
            tp4 = same_close2
            pair2_traded_price = 0.0
            can_take_new_trade = False
    # ── Skipped entries (midline but not armed) ─────────────────────────
    elif ((use_override == 0) and
          ((minutely_low <= middle_prev and p1 == 1) or
           (minutely_low <= middle_line and p1 != 1)) and not long_signal_on):
        tp1 = same_close1
        tp2 = same_close2
        tp3 = same_close1
        tp4 = same_close2
        pair2_traded_price = 0.0
    # ── IN LONG ─────────────────────────────────────────────────────────
    elif long_signal_on == True:
        tal = trade_allocation_1          # trade_allocation[i] = trade_allocation[i-1]
        if (not second_trade_taken) and (sec_off == 0) and (bars_since_entry >= sec_delay_min):
            if p1 == 1:
                sec_trig = minutely_low <= pair1_traded_price * (1 - nbdevupdn * atr_prev / 100)
            else:
                sec_trig = minutely_low <= pair1_traded_price * (1 - nbdevupdn * atr / 100)
            if sec_trig:
                second_trade_taken = True
                if sec_mode == 0:
                    res2 = 1.0
                    traded_price3 = next_close1 * (1 + slippage)
                    traded_price4 = next_close2 * (1 - slippage)
                    tp3 = traded_price3
                    tp4 = traded_price4
                    pair2_traded_price = traded_price3 / traded_price4
                    tcost2 = txn_cost
                    avg_p = (pair1_traded_price + pair2_traded_price) / 2
                    profit_target = 100.0 * ((avg_p * (1 + tp / 100.0)) / pair1_traded_price - 1.0)
                else:
                    if sec_mode == 2:
                        virtual_p2 = (next_close1 * (1 + slippage)) / (next_close2 * (1 - slippage))
                    res2 = 0.0
                    tp3 = same_close1
                    tp4 = same_close2
                    pair2_traded_price = 0.0
            elif False:
                pass
            else:
                res2 = 0.0
                tp3 = same_close1
                tp4 = same_close2
                pair2_traded_price = 0.0
        else:
            if second_trade_taken and sec_mode == 0:
                res2 = 1.0
            tp3 = same_close1
            tp4 = same_close2
            # traded_price3/4 and pair2_traded_price carry (defaults above)

        if second_trade_taken and sec_mode == 0:
            pivot_price = (pair1_traded_price + pair2_traded_price) / 2.0
        elif second_trade_taken and sec_mode == 2:
            pivot_price = (pair1_traded_price + virtual_p2) / 2.0
        else:
            pivot_price = pair1_traded_price

        if upper_eff < pivot_price:
            upper_eff = pivot_price

        if clock_minutes > 0:
            # clock-exit mode: ONLY time-based exit
            if bars_since_entry >= clock_minutes:
                res = 0.0
                res2 = 0.0
                long_signal_on = False
                tp1 = next_close1 * (1 - slippage)
                tp2 = next_close2 * (1 + slippage)
                if second_trade_taken and sec_mode == 0:
                    tp3 = next_close1 * (1 - slippage)
                    tp4 = next_close2 * (1 + slippage)
                    tcost2 = txn_cost
                else:
                    tp3 = same_close1
                    tp4 = same_close2
                second_trade_taken = False
                tcost1 = txn_cost
                exit_reason = 6
            else:
                res = 1.0
                if second_trade_taken:
                    res2 = 1.0
                tp1 = same_close1
                tp2 = same_close2
        else:
            x_eff = x_atr * 1.5 if (sec_mode == 1 and second_trade_taken) else x_atr
            z_atr_hit = qr31_z_atr_stop(z_ema, z_threshold, minutely_low,
                                        minutely_high, pivot_price, x_eff, atr2,
                                        stop_mode, stop_pct)
            corr_hit = corr < corr_threshold

            if z_atr_hit or corr_hit:
                res = 0.0
                res2 = 0.0
                long_signal_on = False
                tp1 = next_close1 * (1 - slippage)
                tp2 = next_close2 * (1 + slippage)
                if second_trade_taken and sec_mode == 0:
                    tp3 = next_close1 * (1 - slippage)
                    tp4 = next_close2 * (1 + slippage)
                    tcost2 = txn_cost
                else:
                    tp3 = same_close1
                    tp4 = same_close2
                second_trade_taken = False
                tcost1 = txn_cost
                can_take_new_trade = False
                bars_since_stop = 0
                if z_atr_hit:
                    exit_reason = 1
                else:
                    exit_reason = 2
            elif ((minutely_high >= upper_eff_prev and p1 == 1) or
                  (minutely_high >= upper_eff and p1 != 1) or
                  (minutely_high > pivot_price * (1 + profit_target / 100))):
                res = 0.0
                res2 = 0.0
                long_signal_on = False
                tp1 = next_close1 * (1 - slippage)
                tp2 = next_close2 * (1 + slippage)
                if second_trade_taken and sec_mode == 0:
                    tp3 = next_close1 * (1 - slippage)
                    tp4 = next_close2 * (1 + slippage)
                    tcost2 = txn_cost
                else:
                    tp3 = same_close1
                    tp4 = same_close2
                second_trade_taken = False
                tcost1 = txn_cost
                if rearm_off == 1:
                    can_take_new_trade = False
                elif ((minutely_high >= upper_eff_prev and p1 == 1) or
                      (minutely_high >= upper_eff and p1 != 1)):
                    can_take_new_trade = True
                else:
                    can_take_new_trade = False
                if minutely_high >= upper_eff_prev and p1 == 1:
                    exit_reason = 3
                elif minutely_high >= upper_eff and p1 != 1:
                    exit_reason = 4
                else:
                    exit_reason = 5
            else:
                res = 1.0
                if second_trade_taken:
                    res2 = 1.0
                tp1 = same_close1
                tp2 = same_close2
    # ── Idle ─────────────────────────────────────────────────────────────
    else:
        long_signal_on = False
        second_trade_taken = False
        tp1 = same_close1
        tp2 = same_close2
        tp3 = same_close1
        tp4 = same_close2
        pair2_traded_price = 0.0
        # traded_price1/2 and pair1_traded_price carry (defaults above)

    ## Direction inversion -- must mirror the array kernel exactly (see its comment).
    if signal_invert == 1:
        res = -res
        res2 = -res2

    return TradeOutputPairLeadLagLastOutput(
        res_lo=res,
        res2_lo=res2,
        tradeprice1_lo=tp1,
        tradeprice2_lo=tp2,
        tradeprice3_lo=tp3,
        tradeprice4_lo=tp4,
        tcost1_lo=tcost1,
        tcost2_lo=tcost2,
        trade_allocation_lo=tal,
        exit_reason_lo=exit_reason,
        long_signal_on=long_signal_on,
        can_take_new_trade=can_take_new_trade,
        second_trade_taken=second_trade_taken,
        pair1_traded_price_lo=pair1_traded_price,
        pair2_traded_price_lo=pair2_traded_price,
        virtual_p2_lo=virtual_p2,
        profit_target_lo=profit_target,
        traded_price1_lo=traded_price1,
        traded_price2_lo=traded_price2,
        traded_price3_lo=traded_price3,
        traded_price4_lo=traded_price4,
        upper_eff_prev_lo=upper_eff,
        bars_since_entry_lo=bars_since_entry,
        bars_since_arm_lo=bars_since_arm,
        bars_since_stop_lo=bars_since_stop,
        same_close1_lo=same_close1,
        same_close2_lo=same_close2,
    )
