"""
Streaming composite z-EMA for v16_v2 reversal2. Mirrors `compute_z_ema` from
`generate_signal_pair_lead_lag.py`:

    ema      = close.ewm(span=ema_len,   adjust=False).mean()
    tr       = max(h-l, |h-prev_c|, |l-prev_c|)
    atr      = tr.ewm(alpha=1/atr_len,   adjust=False).mean()       [Wilder]
    atr_safe = max(atr, 1e-10)
    z        = (close - ema) / atr_safe
    z_ema    = z.ewm(span=z_ema_len,     adjust=False).mean()
"""
import math
from collections import deque

from ._primitives import EWMASpanNoAdjust, WilderATRPrimitive


class ZEMACalc:
    """QR31_v2 note: `ema_last` (the internal EMA of closes) IS the band's middle
    line — the frozen v2 spec makes the centre and the z centre ONE representation
    object (EMA_SPAN=8, adjust=False), so the consumer reads the middle from here
    rather than running a second EMA instance."""

    def __init__(self, ema_len, atr_len, z_ema_len):
        self.ema_len = int(ema_len)
        self.atr_len = int(atr_len)
        self.z_ema_len = int(z_ema_len)
        self._ema = EWMASpanNoAdjust(self.ema_len)
        self._atr = WilderATRPrimitive(self.atr_len)
        self._z_ema = EWMASpanNoAdjust(self.z_ema_len)
        self.z_ema_deque = deque(maxlen=3)
        self.z_ema_last = float("nan")
        self.ema_last = float("nan")      # the middle line (see class docstring)
        self._log = {}

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)
        ema = self._ema.update(c)
        atr = self._atr.update(high, low, close)
        atr_safe = atr if atr >= 1e-10 else 1e-10
        if math.isnan(ema):
            z = float("nan")
        else:
            z = (c - ema) / atr_safe
        self.ema_last = ema
        self.z_ema_last = self._z_ema.update(z)
        self.z_ema_deque.append(self.z_ema_last)
        self._log = {"z_ema_last": self.z_ema_last, "middle_last": self.ema_last}

    @property
    def get_logging_dict(self):
        # Stable latest value (no clear-on-read); see atr_utils.py for rationale.
        return {"z_ema_last": self.z_ema_last, "middle_last": self.ema_last}
