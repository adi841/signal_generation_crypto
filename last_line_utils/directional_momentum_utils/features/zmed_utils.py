"""
Streaming momentum z-score for DMP_v3_2. Mirrors the batch expression in
`get_numba_parameters`:

    zmed = ((close - median20) / atr_eq.clip(lower=1e-12)).ewm(span=zmed_ewm_span,
                                                              adjust=False).mean()

i.e. how far the candle close sits above the channel centre in band-vol units, then
EWMA-smoothed. Both kernels read it in exactly two places, both gates, never sizing —
and the SHORT side is an exact mirror about ZERO, reading the SAME series:

    LONG  entry :  z_median >  EZ            SHORT entry :  z_median < -EZ
    LONG  stop  :  z_median <  Z   and low  < entry_med - X*entry_atr
    SHORT stop  :  z_median > -Z   and high > entry_med + X*entry_atr

At the frozen EZ = 0.0 the two entry gates are exact complements; at Z = -0.3 the two
stop gates arm on |z| > 0.3 in the losing direction. One calc serves both sides.

Note the `adjust` asymmetry against vol_utils: the three vol calcs use adjust=True
(pandas default) while this outer smoothing is adjust=False. That asymmetry is
load-bearing for parity with the frozen book.

SELF-SUFFICIENT by design: this calc owns its own SlidingMedian and EWMSpanStdAdjust
rather than receiving median20/atr_eq piped in per bar. That keeps `_initialize(df)`
needing nothing but OHLC, so the batch warm-up cannot silently disagree with the live
path. The duplicate median/std work is a few float ops per candle. It is deterministic
in the same close stream, so it cannot drift from the standalone RollingMedianCalc /
AtrEqCalc values the kernel receives as `med` / `atr`.

This is NOT reusable from sa_directional's ZMedianCalc, which divides by a Wilder ATR;
DMP divides by atr_eq = close * ewmstd(log-returns), at span 10.
"""
import math
from collections import deque

from ._primitives import SlidingMedian, EWMSpanStdAdjust, EWMASpanNoAdjust

## Floor applied to atr_eq before dividing — mirrors `.clip(lower=1e-12)` in the batch.
ATR_EQ_FLOOR = 1e-12


class ZMedianRvCalc:
    def __init__(self, med_len, atr_span, z_span, z_adjust=False):
        # The frozen config carries zmed_ewm_adjust=false. Assert rather than
        # silently ignore it: an adjust=True smoothing would be a different series.
        assert not z_adjust, (
            "DMP_v3_2 requires zmed_ewm_adjust=False (EWMASpanNoAdjust); "
            f"got z_adjust={z_adjust!r}"
        )
        self.med_len = int(med_len)
        self.atr_span = int(atr_span)
        self.z_span = int(z_span)

        self._sm = SlidingMedian(self.med_len)
        self._std = EWMSpanStdAdjust(self.atr_span)
        self._z = EWMASpanNoAdjust(self.z_span)

        self.prev_close = float("nan")
        self.zmed_deque = deque(maxlen=3)
        self.zmed_last = float("nan")

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        c = float(close)

        # atr_eq leg (same construction as AtrEqCalc)
        if math.isnan(self.prev_close):
            log_r = float("nan")
        else:
            log_r = math.log(c / self.prev_close)
        self.prev_close = c
        sd = self._std.update(log_r)

        # median20 leg
        self._sm.push(c)
        med = self._sm.median()

        # z_raw = (close - median20) / clip(atr_eq, 1e-12, None)
        if math.isnan(med) or math.isnan(sd):
            z_raw = float("nan")
        else:
            atr_eq = c * sd
            denom = atr_eq if atr_eq >= ATR_EQ_FLOOR else ATR_EQ_FLOOR
            z_raw = (c - med) / denom

        # EWMASpanNoAdjust holds the prior value on NaN, matching pandas
        # ewm(adjust=False).mean() over a series with leading/interior NaNs.
        self.zmed_last = self._z.update(z_raw)
        self.zmed_deque.append(self.zmed_last)

    @property
    def get_logging_dict(self):
        return {"zmed_last": self.zmed_last}
