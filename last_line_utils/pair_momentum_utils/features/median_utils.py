"""
Streaming rolling median for B1_v8_v10. Mirrors the batch expression in
`get_numba_parameters`:

    median20 = close.rolling(median_window_bars).median()

This is the Keltner channel CENTRE. The kernel consumes it three ways: as `med`
(the re-arm test `mlow <= med`, and the entry pivot `em = med[i]`), inside
`upper = med + K*atr_eq` (the breakout trigger), and as the base of the entry
sizing z (`zz = (spc-med)/atr`).
"""
from collections import deque

from ._primitives import SlidingMedian


class RollingMedianCalc:
    """
    Streaming `close.rolling(window).median()`.

    Yields NaN until `window` observations have been pushed (matches pandas
    `rolling(window).median()`, which needs a full window by default).
    """

    def __init__(self, window):
        self.window = int(window)
        self._sm = SlidingMedian(self.window)
        self.median_deque = deque(maxlen=3)
        self.median_last = float("nan")

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        self._sm.push(float(close))
        self.median_last = self._sm.median()
        self.median_deque.append(self.median_last)

    @property
    def get_logging_dict(self):
        # Stable latest value (no clear-on-read): log_dump_data runs every minute
        # but _update only fires on candle close, and returning {} on the other
        # minutes makes the per-bucket logging_dict columns ragged.
        return {"median20_last": self.median_last}
