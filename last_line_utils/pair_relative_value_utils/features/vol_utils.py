"""
Streaming EV10 sizing volatility for QR1_v4. Mirrors the batch expression in
`get_numba_parameters` (generate_signal_pair_relative_value.py):

    ev10 = np.log(close).diff().ewm(span=10).std() * 100.0    # adjust=True, % per candle

This is THE candle-fresh input: through the batch's shift(tf-1) projection a
candle's EV10 lands on that candle's own closing minute, and the kernel reads
allocation[i] directly at the entry minute — so an entry decided on a boundary
minute is sized off the candle that closed that same minute. The certification
measured ~0.15 Sharpe (all OOS) for even one candle of staleness: this calc must
be updated the moment the aggregator completes a bar, BEFORE the minute's
allocation is assembled.

The frozen per-cell scale kv and the VT/annf clip stay OUTSIDE the calc, at
allocation-assembly time in update() — matching where the batch applies them.
Structurally a copy of momentum's LongVolCalc (x100 scaling added).
"""
import math
from collections import deque

from ._primitives import EWMSpanStdAdjust


class Ev10Calc:

    def __init__(self, span):
        self.span = int(span)
        self._std = EWMSpanStdAdjust(self.span)
        self.prev_close = float("nan")
        self.ev10_deque = deque(maxlen=3)
        self.ev10_last = float("nan")

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
        sd = self._std.update(log_r)
        self.ev10_last = sd * 100.0 if not math.isnan(sd) else float("nan")
        self.ev10_deque.append(self.ev10_last)

    @property
    def get_logging_dict(self):
        return {"ev10_last": self.ev10_last}
