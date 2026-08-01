"""
Streaming per-LEG features for QR1_v4 — the skew gate and the (dead) corr gate.
Both consume CANDLE-level leg1_close / leg2_close (the two underlying assets,
NOT the pair ratio): warm-up reads the `leg1_close`/`leg2_close` columns the
initialize() warm frame carries; live reads the kwargs update() passes on each
completed bar. Structural ancestors: pair_lead_lag's CorrCalc/SkGateCalc, with
two QR1-critical differences in SkGateCalc (see below).

Batch expressions mirrored (get_numba_parameters):

    corr:  rA, rB = log(cA).diff(), log(cB).diff()          # cA/cB = leg candle closes
           corr   = mean over N in corr_lookbacks of rA.rolling(N).corr(rB)
           (DEAD gate: C_THR = -1; kernel-argument parity only)

    skew:  r  = log(close).diff()
           up = r.where(r > 0, 0.0)      # NOTE: NaN r -> 0.0, not NaN
           dn = (-r).where(r < 0, 0.0)
           upv = sqrt((up**2).rolling(SKEW_N).mean()).clip(lower=1e-10)   # dnv likewise
           sk  = log(upv / dnv)
           sk_gate = ((sk(LEG2) - sk(LEG1)).shift(1) > 0)    # short-flipped orientation

QR1 vs the lead-lag ancestor calc:
  1. ORIENTATION REVERSED: QR1 gates on sk(leg2) - sk(leg1) ("gate TRUE when
     skew(B) > skew(A)" — the short-flipped R1_v16 gate); lead-lag used sk1 - sk2.
  2. NaN RETURNS ARE ZEROED, not poisoned: pandas `.where(cond, 0.0)` sends NaN
     returns to 0.0 (NaN > 0 is False), so the rolling windows keep filling
     through data gaps. The lead-lag calc pushed NaN (poisoning) — that would
     diverge from the QR1 batch at every gap.
"""
import math
from collections import deque

from ._primitives import RollingPearsonCorrPrimitive, RollingFixedWindowSums


class CorrCalc:
    """
    For each N in `lookbacks`, rolling Pearson(log_r_leg1, log_r_leg2) over N.
    `.corr_last` = mean of the N-corrs. QR1 lookbacks = (20, 40, 60, 80).

    NaN propagation: any single window's NaN poisons the mean (matches the
    batch's `csum = csum + cN` where any NaN renders the sum NaN). The batch's
    fillna(1.0) happens at kernel-input assembly, NOT here. Sum-of-products vs
    pandas' mean-centred algorithm can differ ~1e-12 — acceptable: the gate is
    DEAD (C_THR = -1), the value never enters a decision.
    """

    def __init__(self, lookbacks=(20, 40, 60, 80)):
        self.lookbacks = tuple(int(N) for N in lookbacks)
        self._corrs = [RollingPearsonCorrPrimitive(N) for N in self.lookbacks]
        self.prev_close_leg1 = float("nan")
        self.prev_close_leg2 = float("nan")
        self.corr_deque = deque(maxlen=3)
        self.corr_last = float("nan")

    def _initialize(self, df):
        """Batch warm-up. `df` must carry `leg1_close`, `leg2_close` columns."""
        for row in df.itertuples():
            self._update(
                row.Index, row.open, row.high, row.low, row.close,
                leg1_close=row.leg1_close, leg2_close=row.leg2_close,
            )

    def _update(self, curr_time, open_, high, low, close,
                leg1_close=None, leg2_close=None, **kwargs):
        c1 = float(leg1_close) if leg1_close is not None else float("nan")
        c2 = float(leg2_close) if leg2_close is not None else float("nan")
        # positional log-returns: NaN close poisons THIS return and the NEXT
        # (matches np.log(series).diff() row arithmetic).
        if math.isnan(self.prev_close_leg1) or math.isnan(c1):
            r1 = float("nan")
        else:
            r1 = math.log(c1 / self.prev_close_leg1)
        if math.isnan(self.prev_close_leg2) or math.isnan(c2):
            r2 = float("nan")
        else:
            r2 = math.log(c2 / self.prev_close_leg2)
        self.prev_close_leg1 = c1
        self.prev_close_leg2 = c2

        corr_sum = 0.0
        any_nan = False
        for cp in self._corrs:
            cp.push(r1, r2)
            v = cp.corr()
            if math.isnan(v):
                any_nan = True
            else:
                corr_sum += v
        self.corr_last = float("nan") if any_nan else corr_sum / float(len(self._corrs))
        self.corr_deque.append(self.corr_last)

    @property
    def get_logging_dict(self):
        # Stable latest value (no clear-on-read).
        return {"corr_last": self.corr_last}


class SkGateCalc:
    """
    QR1's adverse-skew exclusion gate (SKEW_N = 50).

    `.sk_gate_last` = the most recently EMITTED gate — i.e. the shift(1) of the
    underlying sk(leg2) - sk(leg1) diff: on each completed bar the gate is taken
    from the PREVIOUS bar's diff, then the new diff is computed for the next bar
    to emit. NaN diff -> 0.0 (NaN > 0 is False in pandas). Removing this gate
    triples the book MaxDD — orientation errors here are catastrophic and the
    parity test pins it against the batch series.
    """

    def __init__(self, skN=50):
        self.skN = int(skN)
        self._up1 = RollingFixedWindowSums(self.skN)
        self._dn1 = RollingFixedWindowSums(self.skN)
        self._up2 = RollingFixedWindowSums(self.skN)
        self._dn2 = RollingFixedWindowSums(self.skN)
        self.prev_close_leg1 = float("nan")
        self.prev_close_leg2 = float("nan")
        self.sk_gate_last = 0.0   # default (NaN > 0 -> False -> 0.0)
        self._pending_sk_diff = float("nan")   # the unshifted "current" diff
        self.sk_gate_deque = deque(maxlen=3)

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(
                row.Index, row.open, row.high, row.low, row.close,
                leg1_close=row.leg1_close, leg2_close=row.leg2_close,
            )

    def _update(self, curr_time, open_, high, low, close,
                leg1_close=None, leg2_close=None, **kwargs):
        c1 = float(leg1_close) if leg1_close is not None else float("nan")
        c2 = float(leg2_close) if leg2_close is not None else float("nan")
        if math.isnan(self.prev_close_leg1) or math.isnan(c1):
            r1 = float("nan")
        else:
            r1 = math.log(c1 / self.prev_close_leg1)
        if math.isnan(self.prev_close_leg2) or math.isnan(c2):
            r2 = float("nan")
        else:
            r2 = math.log(c2 / self.prev_close_leg2)
        self.prev_close_leg1 = c1
        self.prev_close_leg2 = c2

        # 1. emit the SHIFTED gate (previous bar's diff)
        prev_diff = self._pending_sk_diff
        self.sk_gate_last = 1.0 if (not math.isnan(prev_diff) and prev_diff > 0.0) else 0.0
        self.sk_gate_deque.append(self.sk_gate_last)

        # 2. update the rolling means of squared up/down moves. `.where(cond, 0.0)`
        #    semantics: a NaN return contributes 0.0 (windows fill through gaps).
        if math.isnan(r1):
            u1s, d1s = 0.0, 0.0
        else:
            u1 = r1 if r1 > 0.0 else 0.0
            d1 = -r1 if r1 < 0.0 else 0.0
            u1s, d1s = u1 * u1, d1 * d1
        if math.isnan(r2):
            u2s, d2s = 0.0, 0.0
        else:
            u2 = r2 if r2 > 0.0 else 0.0
            d2 = -r2 if r2 < 0.0 else 0.0
            u2s, d2s = u2 * u2, d2 * d2
        self._up1.push(u1s)
        self._dn1.push(d1s)
        self._up2.push(u2s)
        self._dn2.push(d2s)

        # 3. the new diff (emitted NEXT bar): sk(leg2) - sk(leg1) — short-flipped.
        m_up1, m_dn1 = self._up1.mean(), self._dn1.mean()
        m_up2, m_dn2 = self._up2.mean(), self._dn2.mean()
        if math.isnan(m_up1) or math.isnan(m_dn1) or math.isnan(m_up2) or math.isnan(m_dn2):
            self._pending_sk_diff = float("nan")
        else:
            upv1 = max(math.sqrt(m_up1), 1e-10)
            dnv1 = max(math.sqrt(m_dn1), 1e-10)
            upv2 = max(math.sqrt(m_up2), 1e-10)
            dnv2 = max(math.sqrt(m_dn2), 1e-10)
            sk1 = math.log(upv1 / dnv1)
            sk2 = math.log(upv2 / dnv2)
            self._pending_sk_diff = sk2 - sk1

    @property
    def get_logging_dict(self):
        # Stable latest value (no clear-on-read).
        return {"sk_gate_last": self.sk_gate_last}
