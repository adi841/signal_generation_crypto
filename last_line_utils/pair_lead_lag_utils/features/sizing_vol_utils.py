"""
Streaming QR31_v2 sizing volatility. Mirrors the batch chain in
`generate_signal_pair_lead_lag.py::get_numba_parameters`:

    r      = log(close).diff()                                   [candle closes]
    v      = sqrt( (r**2).ewm(span=span, adjust=False).mean() ) * 100
    v      = v * scale                    [IS-frozen scale: mean(clipped ATR75 %)
                                           / mean(raw EWMA75 %) over the IS window]
    v      = clip(v, lo, hi)              [IS-frozen bounds of the SCALED series]

scale / lo / hi come from the frozen per-cell artifact
(qr31_cell_constants.parquet) via strategy_config — they are part of the frozen
parameterisation and are never recomputed on live data.

The allocation itself — clip((VT/annf)/sizing_vol, 0, 1), x the daily R x I
multiplier when its artifact lands — is assembled at the consumer
(pair_lead_lag_last_line_utils.update), not here.
"""
import math
from collections import deque

from ._primitives import EWMASpanNoAdjust


class SizingVolCalc:
    def __init__(self, span, scale, lo_cutoff, hi_cutoff):
        self.span = int(span)
        self.scale = float(scale)
        self.lo_cutoff = float(lo_cutoff)
        self.hi_cutoff = float(hi_cutoff)
        self._ewm = EWMASpanNoAdjust(self.span)
        self.prev_log_close = float("nan")
        self.sizing_vol_deque = deque(maxlen=3)
        self.sizing_vol_last = float("nan")
        self._log = {}

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)
        # log-return, NaN at the first bar (pandas .diff() semantics). Computed as
        # log(c) - log(prev_c) — NOT log(c/prev_c) — matching the batch's
        # `np.log(close).diff()` construction. NOTE the achievable parity class:
        # numpy's Series-level np.log is a SIMD implementation whose LAST BIT can
        # differ from scalar libm on some inputs, so this chain matches the batch to
        # ~1 ULP of the return (≲1e-13 relative on the vol), not bitwise — the same
        # tolerance class the matching tests use for every EWM-derived column.
        log_c = math.log(c)
        if math.isnan(self.prev_log_close):
            r = float("nan")
        else:
            r = log_c - self.prev_log_close
        self.prev_log_close = log_c

        m = self._ewm.update(r * r if not math.isnan(r) else float("nan"))
        if math.isnan(m):
            v = float("nan")
        else:
            v = math.sqrt(m) * 100.0 * self.scale
            if v < self.lo_cutoff:
                v = self.lo_cutoff
            elif v > self.hi_cutoff:
                v = self.hi_cutoff
        self.sizing_vol_last = v
        self.sizing_vol_deque.append(v)
        self._log = {"sizing_vol_last": v}

    @property
    def get_logging_dict(self):
        # Stable latest value (no clear-on-read); see atr_utils.py for rationale.
        return {"sizing_vol_last": self.sizing_vol_last}
