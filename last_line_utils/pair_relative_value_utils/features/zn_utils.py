"""
Streaming zn for QR1_v4 — the v4 entry AND exit z. Mirrors the batch expression
in `get_numba_parameters` (generate_signal_pair_relative_value.py):

    lr      = np.log(close).diff()
    zn_raw  = (close - close.ewm(span=10, adjust=False).mean())
              / (lr.ewm(span=14).std() * close).clip(lower=1e-12)
    zn      = zn_raw.ewm(span=20, adjust=False).mean()

The adjust asymmetry is LOAD-BEARING: both zn EMAs are adjust=False while the
band centre EMA10 (band_utils) is adjust=True, and the vol leg is adjust=True
debiased. This is NOT prep_cell's ATR-based z_ema — the v4 engine replaced it.

SELF-SUFFICIENT by design (momentum convention): owns its own EMA/std primitives
and consumes only OHLC, so warm-up and live cannot silently disagree.
"""
import math
from collections import deque

from ._primitives import EWMASpanNoAdjust, EWMSpanStdAdjust


class ZnCalc:

    def __init__(self, centre_span, vol_span, smooth_span, denom_floor):
        self._centre = EWMASpanNoAdjust(int(centre_span))
        self._vol = EWMSpanStdAdjust(int(vol_span))
        self._smooth = EWMASpanNoAdjust(int(smooth_span))
        self.denom_floor = float(denom_floor)
        self.prev_close = float("nan")
        self.zn_last = float("nan")
        self.zn_deque = deque(maxlen=3)

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)
        if math.isnan(self.prev_close):
            log_r = float("nan")
        else:
            log_r = math.log(c / self.prev_close)
        self.prev_close = c

        centre = self._centre.update(c)
        vol = self._vol.update(log_r)

        if math.isnan(centre) or math.isnan(vol):
            zn_raw = float("nan")
        else:
            # (vol * close).clip(lower=denom_floor) — floor the PRICE-UNIT vol
            denom = vol * c
            if denom < self.denom_floor:
                denom = self.denom_floor
            zn_raw = (c - centre) / denom

        # EWMASpanNoAdjust holds the prior value on NaN, matching pandas
        # ewm(adjust=False).mean() over a series with leading NaNs.
        self.zn_last = self._smooth.update(zn_raw)
        self.zn_deque.append(self.zn_last)

    @property
    def get_logging_dict(self):
        return {"zn_last": self.zn_last}
