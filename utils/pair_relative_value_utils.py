import numpy as np
from numba import jit
from collections import namedtuple

def qr1_bandwidth_gate(f_bandwp, wmed, atr_pct, atr2_pct, corr, c_thr, gate_mode):
    """The QR1_v4 band-width ENTRY gate, in ONE place.

    Shared by the batch signal builder (generate_signal_pair_relative_value.py) and the
    live per-minute path (pair_relative_value_last_line_utils.py) so the two cannot drift
    apart. Unlike the other three sleeves this factor lives OUTSIDE the numba kernel -- it
    is one conjunct of `msig`, which the kernel then consumes as a single +/-1 array.

    The factor study measured it as this sleeve's largest contributor (-0.95 Sharpe when
    removed), and the kernel's own breach instrumentation shows the sleeve rejects 99.6%
    of the setups it sees -- the entry gates do nearly all the work here.

      gate_mode 0  PRODUCTION      f_bandwp >  wmed    (wide-band regime only)
                1  INVERTED        f_bandwp <  wmed    trades exactly the narrow-band
                                                       regime the strategy exists to avoid
                2  OFF             always True         (ablation)
                3  VOL-EXPANSION   atr_pct > atr2_pct  fast ATR above slow ATR
                4  CORRELATION     corr > c_thr        revives the gate that is dead in
                                                       production (C_THR = -1.0)

    Works on scalars (live) and arrays (batch) alike. NaN compares False throughout, which
    matches the production semantics of both paths.
    """
    if gate_mode == 1:
        return np.less(f_bandwp, wmed)
    if gate_mode == 2:
        return np.full(np.shape(f_bandwp), True)
    if gate_mode == 3:
        return np.greater(atr_pct, atr2_pct)
    if gate_mode == 4:
        return np.greater(corr, c_thr)
    return np.greater(f_bandwp, wmed)


## QR1_v4 (Pairs Relative Value) trade kernel outputs. DUAL-tranche SHORT stream on the
## leg1/leg2 ratio (short ratio = -leg1, +leg2, dollar-neutral): res / res2 (-1/0 per
## tranche), per-leg trade prices per tranche (tranche 1 fills tp1/tp2, scale-in tranche
## fills tp3/tp4), per-tranche per-bar trade cost, and the trade allocation (tal —
## captured ONCE at the FIRST entry and shared by the scale-in; carries between bars and
## is NOT reset on exit, cost scaling uses it). Each tranche is HALF the cell allocation:
## minute position = (res + res2) * tal * 0.5 on the ratio.
##
## Unlike B1 (single stream), the reversal2/QR31 dual-stream contract (res2 /
## tradeprice3,4 / order_tag2) DOES apply to this sleeve.
TradeOutputPairRelativeValueArray = namedtuple("TradeOutputArray", [
    "res_arr",
    "res2_arr",
    "tradeprice1_arr",
    "tradeprice2_arr",
    "tradeprice3_arr",
    "tradeprice4_arr",
    "tcost1_arr",
    "tcost2_arr",
    "trade_allocation_arr",
])

TradeOutputPairRelativeValueLastOutput = namedtuple("TradeOutputLastOutput", [
    "res_lo",
    "res2_lo",
    "tradeprice1_lo",
    "tradeprice2_lo",
    "tradeprice3_lo",
    "tradeprice4_lo",
    "tcost1_lo",
    "tcost2_lo",
    "trade_allocation_lo",
    # carried kernel state — everything the per-bar _ll kernel needs to resume from the
    # batch warm-up. The kernel reads middle/upper[i-1] only via the p1 convention (the
    # live layer supplies previous-candle lines), so this is pure trade state.
    # NOTE: the frozen reference kernel also tracks tsm / first_breach / cur_e1 / cur_e2 —
    # those feed ONLY its event-log instrumentation, never a decision, and are not
    # carried here (bitwise output equivalence vs the frozen kernel is asserted in the
    # step's harness). tsm for the SCORE feature is computed outside the kernel.
    "short_signal_on",       # short_on  (tranche 1 open)
    "second_signal_on",      # second_on (scale-in tranche open)
    "can_take_new_trade",    # can_new   (re-armed by a middle touch; cleared by stops)
    "entry_pair_price",      # pair1_traded_price_1st (entry-1 ratio fill — stop & scale-in pivot)
    "profit_target_lo",      # profit_target1 (vestigial: written at entry, never read)
    # live-layer conveniences (DB price reporting):
    "same_close1_lo",
    "same_close2_lo",
])


class TradeOutputPairRelativeValueLastOutputCls:
    """Mutable mirror of TradeOutputPairRelativeValueLastOutput, held on the strategy
    object as `numba_cls` and splatted back into the per-bar kernel each minute. The
    defaults are the array kernel's own initial values, so a cold start and a warm
    resume agree."""

    def __init__(self, res_lo, res2_lo, tradeprice1_lo, tradeprice2_lo,
                 tradeprice3_lo, tradeprice4_lo, tcost1_lo, tcost2_lo,
                 trade_allocation_lo,
                 short_signal_on=False, second_signal_on=False, can_take_new_trade=True,
                 entry_pair_price=0.0, profit_target_lo=0.0,
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
        self.short_signal_on = short_signal_on
        self.second_signal_on = second_signal_on
        self.can_take_new_trade = can_take_new_trade
        self.entry_pair_price = entry_pair_price
        self.profit_target_lo = profit_target_lo
        self.same_close1_lo = same_close1_lo
        self.same_close2_lo = same_close2_lo

    def __repr__(self) -> str:
        return (f"TradeOutputPairRelativeValueLastOutputCls(res_lo={self.res_lo}, "
                f"res2_lo={self.res2_lo}, tradeprice1_lo={self.tradeprice1_lo}, "
                f"tradeprice2_lo={self.tradeprice2_lo}, tradeprice3_lo={self.tradeprice3_lo}, "
                f"tradeprice4_lo={self.tradeprice4_lo}, tcost1_lo={self.tcost1_lo}, "
                f"tcost2_lo={self.tcost2_lo}, trade_allocation_lo={self.trade_allocation_lo}, "
                f"short_signal_on={self.short_signal_on}, second_signal_on={self.second_signal_on}, "
                f"can_take_new_trade={self.can_take_new_trade}, entry_pair_price={self.entry_pair_price}, "
                f"profit_target_lo={self.profit_target_lo}, same_close1_lo={self.same_close1_lo}, "
                f"same_close2_lo={self.same_close2_lo})")


@jit(nopython=True)
def cryptopairs_qr1v4_short_iact(next_close1, same_close1, tp,
                                 minutely_high, minutely_low, p1,
                                 next_close2, same_close2,
                                 txn_cost, m1, m2,
                                 upper_line, middle_line, lower_line,
                                 atr, atr2, second_entry_atr_mult,
                                 slippage, allocation,
                                 z_ema, z_threshold, x_atr,
                                 corr, corr_threshold,
                                 z_entry, skew_ok, signal_invert=0):
    """QR1_v4 kernel: SHORT relative-value fade on the leg1/leg2 price ratio, evaluated
    on the minutely frame with p1 marking tf-bar closes. Behavioural transcription of
    the frozen reference `qr1_instr` (crypto_sims/PRODUCTION/engines/core/kernel_qr1.py)
    with its event-log instrumentation stripped — bitwise output equivalence vs the
    frozen kernel is asserted in this step's harness. The trailing (z_entry, skew_ok)
    arguments are the reference's decomposition-only inputs, kept for signature parity;
    they never enter a decision (msig already folds all three entry gates).

    THE KERNEL IS THE SPECIFICATION. Where config comments or writeups disagree with
    the body below, the body wins. In particular:
      * Active lines follow the p1 convention: on a candle-boundary minute (p1 == 1)
        the middle/upper used intrabar are the PREVIOUS row's (the fresh candle was not
        closed while the minute traded). allocation[i] is read directly — an entry on a
        boundary minute is sized off the just-closed candle (candle-fresh EV10).
      * Re-arm: minutely_low <= active middle sets can_new. The z/ATR stop CLEARS
        can_new (no re-entry until the next middle touch); the profit exit sets it.
      * ENTRY 1 (flat, breach of active upper, armed, msig regime ok — the engine
        passes m2 = 1 so entry requires m1[i] == +1): res = -1, fills at
        next1*(1-slip)/next2*(1+slip), tal = allocation[i] captured for the trade.
      * SCALE-IN: minutely_high >= entry-1 ratio * (1 + SEC_MULT*atr/100) with msig ok
        -> res2 = -1 at tp3/tp4; it REUSES entry-1's tal (sizing is not re-read).
      * EXITS (shared by both tranches): z/ATR stop = z > z_threshold AND minutely_high
        >= entry-1 ratio * (1 + x_atr*atr2/100) -> can_new = False; corr exit is DEAD
        in v4 (corr_threshold = -1, corr in [-1, 1] can never be below it); profit =
        minutely_low <= active middle -> can_new = True. A both-tranche exit fills both
        at the same prices; profit-only sets can_new, any stop involvement clears it.
      * tal is written ONLY on the entry-1 bar and carries forward unchanged —
        deliberately not reset on exit, because the exits' cost scaling still needs it.
      * tp (profit_target) is vestigial: written into profit_target1 at entry, never
        read (the reference feeds a constant 1000.0).

    Returns (TradeOutputPairRelativeValueArray, TradeOutputPairRelativeValueLastOutput)."""
    n = next_close1.shape[0]
    res = np.zeros(n)
    res2 = np.zeros(n)
    tradeprice1 = np.empty(n); tradeprice2 = np.empty(n)
    tradeprice3 = np.empty(n); tradeprice4 = np.empty(n)
    total_tc1 = np.zeros(n); total_tc2 = np.zeros(n)
    trade_allocation = np.zeros(n)

    pair1_traded_price_1st = np.zeros(n)
    profit_target1 = np.zeros(n)

    tradeprice1[0] = same_close1[0]; tradeprice2[0] = same_close2[0]
    tradeprice3[0] = same_close1[0]; tradeprice4[0] = same_close2[0]

    short_on = False
    second_on = False
    can_new = True

    for i in range(1, n):
        pair1_traded_price_1st[i] = pair1_traded_price_1st[i - 1]
        profit_target1[i] = profit_target1[i - 1]
        trade_allocation[i] = trade_allocation[i - 1]

        # active middle/upper per the p1 convention
        if p1[i] == 1:
            mid_act = middle_line[i - 1]
            up_act = upper_line[i - 1]
        else:
            mid_act = middle_line[i]
            up_act = upper_line[i]

        # 1. Re-arm on middle touch
        if minutely_low[i] <= mid_act:
            can_new = True

        # 2. breach flag
        spread_above = minutely_high[i] >= up_act

        # 3. State machine
        if (not short_on) and spread_above and can_new:
            if (m1[i] == -1 and m2 == -1) or (m1[i] == 1 and m2 == 1):
                short_on = True
                second_on = False
                traded1 = next_close1[i] * (1 - slippage)
                traded2 = next_close2[i] * (1 + slippage)
                tradeprice1[i] = traded1
                tradeprice2[i] = traded2
                tradeprice3[i] = same_close1[i]
                tradeprice4[i] = same_close2[i]
                pair1_traded_price_1st[i] = traded1 / traded2
                res[i] = -1
                profit_target1[i] = tp[i]
                total_tc1[i] = txn_cost
                trade_allocation[i] = allocation[i]
            else:
                tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
                tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]

        elif short_on and not second_on:
            z_atr_hit = (z_ema[i] > z_threshold) and (
                minutely_high[i] >= pair1_traded_price_1st[i] * (1 + x_atr * atr2[i] / 100)
            )
            corr_hit = corr[i] < corr_threshold

            if p1[i] == 1:
                profit_hit = minutely_low[i] <= middle_line[i - 1]
            else:
                profit_hit = minutely_low[i] <= middle_line[i]

            if z_atr_hit or corr_hit:
                res[i] = 0
                short_on = False
                tradeprice1[i] = next_close1[i] * (1 + slippage)
                tradeprice2[i] = next_close2[i] * (1 - slippage)
                tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]
                total_tc1[i] = txn_cost
                can_new = False
            elif profit_hit:
                res[i] = 0
                short_on = False
                tradeprice1[i] = next_close1[i] * (1 + slippage)
                tradeprice2[i] = next_close2[i] * (1 - slippage)
                tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]
                total_tc1[i] = txn_cost
                can_new = True
            else:
                second_trigger = (
                    minutely_high[i]
                    >= pair1_traded_price_1st[i] * (1 + second_entry_atr_mult * atr[i] / 100)
                )
                regime_ok = (m1[i] == -1 and m2 == -1) or (m1[i] == 1 and m2 == 1)
                if second_trigger and regime_ok:
                    second_on = True
                    t2a = next_close1[i] * (1 - slippage)
                    t2b = next_close2[i] * (1 + slippage)
                    res[i] = -1
                    res2[i] = -1
                    tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
                    tradeprice3[i] = t2a
                    tradeprice4[i] = t2b
                    total_tc2[i] = txn_cost
                else:
                    res[i] = -1
                    tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
                    tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]

        elif short_on and second_on:
            z_atr_hit = (z_ema[i] > z_threshold) and (
                minutely_high[i] >= pair1_traded_price_1st[i] * (1 + x_atr * atr2[i] / 100)
            )
            corr_hit = corr[i] < corr_threshold
            if p1[i] == 1:
                profit_hit = minutely_low[i] <= middle_line[i - 1]
            else:
                profit_hit = minutely_low[i] <= middle_line[i]

            if z_atr_hit or corr_hit or profit_hit:
                res[i] = 0
                res2[i] = 0
                short_on = False
                second_on = False
                tradeprice1[i] = next_close1[i] * (1 + slippage)
                tradeprice2[i] = next_close2[i] * (1 - slippage)
                tradeprice3[i] = next_close1[i] * (1 + slippage)
                tradeprice4[i] = next_close2[i] * (1 - slippage)
                total_tc1[i] = txn_cost
                total_tc2[i] = txn_cost
                if profit_hit and not (z_atr_hit or corr_hit):
                    can_new = True
                else:
                    can_new = False
            else:
                res[i] = -1
                res2[i] = -1
                tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
                tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]

        else:
            short_on = False
            second_on = False
            tradeprice1[i] = same_close1[i]; tradeprice2[i] = same_close2[i]
            tradeprice3[i] = same_close1[i]; tradeprice4[i] = same_close2[i]

    ####
    ## Direction inversion -- trade AGAINST this sleeve's own signal. Applied to the
    ## REPORTED position only, after the state machine has run. Both tranches flip.
    ## QR1 is short-only, so inverting makes it a long relative-value sleeve.
    if signal_invert == 1:
        res = -res
        res2 = -res2
    trade_output_array = TradeOutputPairRelativeValueArray(
        res, res2, tradeprice1, tradeprice2, tradeprice3, tradeprice4,
        total_tc1, total_tc2, trade_allocation)
    trade_output_last_output = TradeOutputPairRelativeValueLastOutput(
        res_lo=res[-1],
        res2_lo=res2[-1],
        tradeprice1_lo=tradeprice1[-1],
        tradeprice2_lo=tradeprice2[-1],
        tradeprice3_lo=tradeprice3[-1],
        tradeprice4_lo=tradeprice4[-1],
        tcost1_lo=total_tc1[-1],
        tcost2_lo=total_tc2[-1],
        trade_allocation_lo=trade_allocation[-1],
        short_signal_on=short_on,
        second_signal_on=second_on,
        can_take_new_trade=can_new,
        entry_pair_price=pair1_traded_price_1st[-1],
        profit_target_lo=profit_target1[-1],
        same_close1_lo=same_close1[-1],
        same_close2_lo=same_close2[-1],
    )
    return trade_output_array, trade_output_last_output


@jit(nopython=True)
def cryptopairs_qr1v4_short_iact_ll(next_close1, same_close1, tp,
                                    minutely_high, minutely_low, p1,
                                    next_close2, same_close2,
                                    txn_cost, m1, m2,
                                    upper_line, middle_line, prev_middle_line, prev_upper_line,
                                    atr, atr2, second_entry_atr_mult,
                                    slippage, allocation,
                                    z_ema, z_threshold, x_atr,
                                    corr, corr_threshold,
                                    trade_allocation_1,
                                    short_signal_on, second_signal_on, can_take_new_trade,
                                    entry_pair_price, profit_target_1, signal_invert=0):
    """Per-bar (last-line) form of cryptopairs_qr1v4_short_iact: the same per-bar
    quantities as the array kernel but as SCALARS (with prev_middle_line /
    prev_upper_line supplied explicitly, since the p1==1 convention reads the previous
    row's lines and a per-bar call has no [i-1]; lower_line is dropped — the short
    kernel never reads it), then the carried decision state in
    TradeOutputPairRelativeValueLastOutput order (the last-value fields are re-derived
    per bar). Returns a single TradeOutputPairRelativeValueLastOutput which becomes the
    next call's state. Bar-by-bar state-threaded equivalence vs the array kernel is
    asserted bitwise in the step's harness.

    Live-layer notes: `next_close1/2` are the live fill approximations (HLC3 — live
    cannot see the batch's forward OHLC4 mean); `entry_pair_price` must be a float by
    call time (live.py's sentinel backfill runs before the kernel)."""
    res = 0.0
    res2 = 0.0
    tradeprice1 = same_close1
    tradeprice2 = same_close2
    tradeprice3 = same_close1
    tradeprice4 = same_close2
    total_tc1 = 0.0
    total_tc2 = 0.0
    trade_allocation = trade_allocation_1

    short_on = short_signal_on
    second_on = second_signal_on
    can_new = can_take_new_trade
    epp = entry_pair_price
    pt = profit_target_1

    # active middle/upper per the p1 convention (prev lines supplied explicitly)
    if p1 == 1:
        mid_act = prev_middle_line
        up_act = prev_upper_line
    else:
        mid_act = middle_line
        up_act = upper_line

    # 1. Re-arm on middle touch
    if minutely_low <= mid_act:
        can_new = True

    # 2. breach flag
    spread_above = minutely_high >= up_act

    # 3. State machine (scalar transcription of the array kernel's branches)
    if (not short_on) and spread_above and can_new:
        if (m1 == -1 and m2 == -1) or (m1 == 1 and m2 == 1):
            short_on = True
            second_on = False
            traded1 = next_close1 * (1 - slippage)
            traded2 = next_close2 * (1 + slippage)
            tradeprice1 = traded1
            tradeprice2 = traded2
            tradeprice3 = same_close1
            tradeprice4 = same_close2
            epp = traded1 / traded2
            res = -1.0
            pt = tp
            total_tc1 = txn_cost
            trade_allocation = allocation

    elif short_on and not second_on:
        z_atr_hit = (z_ema > z_threshold) and (
            minutely_high >= epp * (1 + x_atr * atr2 / 100)
        )
        corr_hit = corr < corr_threshold
        profit_hit = minutely_low <= mid_act

        if z_atr_hit or corr_hit:
            res = 0.0
            short_on = False
            tradeprice1 = next_close1 * (1 + slippage)
            tradeprice2 = next_close2 * (1 - slippage)
            total_tc1 = txn_cost
            can_new = False
        elif profit_hit:
            res = 0.0
            short_on = False
            tradeprice1 = next_close1 * (1 + slippage)
            tradeprice2 = next_close2 * (1 - slippage)
            total_tc1 = txn_cost
            can_new = True
        else:
            second_trigger = (
                minutely_high >= epp * (1 + second_entry_atr_mult * atr / 100)
            )
            regime_ok = (m1 == -1 and m2 == -1) or (m1 == 1 and m2 == 1)
            if second_trigger and regime_ok:
                second_on = True
                res = -1.0
                res2 = -1.0
                tradeprice3 = next_close1 * (1 - slippage)
                tradeprice4 = next_close2 * (1 + slippage)
                total_tc2 = txn_cost
            else:
                res = -1.0

    elif short_on and second_on:
        z_atr_hit = (z_ema > z_threshold) and (
            minutely_high >= epp * (1 + x_atr * atr2 / 100)
        )
        corr_hit = corr < corr_threshold
        profit_hit = minutely_low <= mid_act

        if z_atr_hit or corr_hit or profit_hit:
            res = 0.0
            res2 = 0.0
            short_on = False
            second_on = False
            tradeprice1 = next_close1 * (1 + slippage)
            tradeprice2 = next_close2 * (1 - slippage)
            tradeprice3 = next_close1 * (1 + slippage)
            tradeprice4 = next_close2 * (1 - slippage)
            total_tc1 = txn_cost
            total_tc2 = txn_cost
            if profit_hit and not (z_atr_hit or corr_hit):
                can_new = True
            else:
                can_new = False
        else:
            res = -1.0
            res2 = -1.0

    else:
        short_on = False
        second_on = False

    ## Direction inversion -- must mirror the array kernel exactly (see its comment).
    if signal_invert == 1:
        res = -res
        res2 = -res2

    return TradeOutputPairRelativeValueLastOutput(
        res_lo=res,
        res2_lo=res2,
        tradeprice1_lo=tradeprice1,
        tradeprice2_lo=tradeprice2,
        tradeprice3_lo=tradeprice3,
        tradeprice4_lo=tradeprice4,
        tcost1_lo=total_tc1,
        tcost2_lo=total_tc2,
        trade_allocation_lo=trade_allocation,
        short_signal_on=short_on,
        second_signal_on=second_on,
        can_take_new_trade=can_new,
        entry_pair_price=epp,
        profit_target_lo=pt,
        same_close1_lo=same_close1,
        same_close2_lo=same_close2,
    )
