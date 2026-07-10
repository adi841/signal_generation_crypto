"""
Streaming per-leg corr feature:
  - CorrCalc — multi-window mean of rolling Pearson(log_r_leg1, log_r_leg2).

Consumes CANDLE-level leg1_close / leg2_close (the two underlying assets, NOT the
pair ratio). NaN-poisoning matches the batch path: NaN until N consecutive valid
pairs have been observed for the longest window.

QR31_v2 uses this both as the corr-EXIT input (corr < c_thr per TF) and, in
ablation-neutral form, as the disabled entry gate. The retired v16_v2 SkGateCalc
(skew-asymmetry gate) was deleted with the v16_v2 machine — QR31_v2 has no skew
gate; QR1's lives in pair_relative_value_utils/features/corr_utils.py.
"""
import math
from collections import deque

from ._primitives import RollingPearsonCorrPrimitive


class CorrCalc:
    """
    For each N in `lookbacks`, rolling Pearson(log_r_leg1, log_r_leg2) over N.
    `.corr_last = mean of the N-corrs`. Driver default lookbacks = (20, 40, 60, 80).

    NaN propagation: any single window's NaN poisons the sum (matches the
    batch's `corr_sum += corr_N` where any NaN renders the cumulative sum NaN).
    """

    def __init__(self, lookbacks=(20, 40, 60, 80)):
        self.lookbacks = tuple(int(N) for N in lookbacks)
        self._corrs = [RollingPearsonCorrPrimitive(N) for N in self.lookbacks]
        self.prev_log_close_leg1 = float("nan")
        self.prev_log_close_leg2 = float("nan")
        self.corr_deque = deque(maxlen=3)
        self.corr_last = float("nan")
        self._log = {}

    def _initialize(self, df):
        """Batch warm-up. `df` must carry `leg1_close`, `leg2_close` columns."""
        for row in df.itertuples():
            self._update(
                row.Index, row.open, row.high, row.low, row.close,
                leg1_close=row.leg1_close, leg2_close=row.leg2_close,
            )

    def _update(self, curr_time, open_, high, low, close,
                leg1_close=None, leg2_close=None, **kwargs):
        c1 = float(leg1_close)
        c2 = float(leg2_close)
        # log-returns, NaN at the first bar. Computed as log(c) - log(prev_c) — NOT
        # log(c/prev_c) — matching the batch's `np.log(closes).diff()` construction.
        # (Parity class: ~1 ULP — numpy's SIMD log vs scalar libm; plus running-sum
        # vs pandas rolling summation order. Tolerance column in the matching tests.)
        log_c1 = math.log(c1)
        log_c2 = math.log(c2)
        if math.isnan(self.prev_log_close_leg1):
            r1 = float("nan")
        else:
            r1 = log_c1 - self.prev_log_close_leg1
        if math.isnan(self.prev_log_close_leg2):
            r2 = float("nan")
        else:
            r2 = log_c2 - self.prev_log_close_leg2
        self.prev_log_close_leg1 = log_c1
        self.prev_log_close_leg2 = log_c2

        corr_sum = 0.0
        any_nan = False
        for cp in self._corrs:
            cp.push(r1, r2)
            v = cp.corr()
            if math.isnan(v):
                any_nan = True
            corr_sum += v if not math.isnan(v) else float("nan")
        if any_nan:
            self.corr_last = float("nan")
        else:
            self.corr_last = corr_sum / float(len(self._corrs))
        self.corr_deque.append(self.corr_last)
        self._log = {"corr_last": self.corr_last}

    @property
    def get_logging_dict(self):
        # Stable latest value (no clear-on-read); see atr_utils.py for rationale.
        return {"corr_last": self.corr_last}

