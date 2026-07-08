"""
Streaming rolling median for DMP_v3_2. Mirrors the batch expression in
`get_numba_parameters`:

    median20 = close.rolling(median_window_bars).median()

This is the Keltner channel CENTRE, and it is the single most load-bearing feature in
the strategy — every kernel branch touches it. Both kernels consume it three ways:

    re-arm    LONG  `minutely_low  <= median_line` / SHORT `minutely_high >= median_line`
    entry     LONG  `asset_close > median_line + K*atr_eq`
              SHORT `asset_close < median_line - K*atr_eq`
              and it is latched as the entry pivot `entry_med = median_line[i]`
    sizing    `zz = (asset_close - median_line)/atr_eq`  (SHORT: numerator reversed)

Note the hard stop measures from `entry_med` — the channel centre AT ENTRY — not from
the entry price, which is why the pivot is latched rather than re-read.
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
