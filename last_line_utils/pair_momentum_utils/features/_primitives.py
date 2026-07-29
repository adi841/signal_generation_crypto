"""
Streaming primitives for the B1_v8_v10 pair-momentum features.

Each primitive is scalar-state, advances one observation per `update(...)` /
`push(...)` call, and matches the semantics of the corresponding pandas expression
in `get_numba_parameters` (generate_signal_pair_momentum.py).

These are copied VERBATIM from `last_line_utils/sa_directional_utils/features/
_primitives.py` (SlidingMedian :19, EWMASpanNoAdjust :108, EWMSpanStdAdjust :132),
following the per-sleeve convention that every `features/` package owns its own
primitives rather than importing across sleeves. Only the four B1 needs are copied;
WilderATRPrimitive and the tile helper are not used here — B1's `atr_eq` is
`close * ewmstd(log-returns)`, NOT a Wilder ATR.
"""
import heapq
import math
from collections import deque

import numpy as np


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


class SlidingMedian:
    """
    Fixed-window rolling median using two heaps with lazy deletion.

    `push(value)` adds a new observation; once `window` items have been pushed,
    each subsequent push also evicts the oldest. `median()` returns NaN until
    the window is full (matches `close.rolling(window).median()`).

    Invariant: after `_balance`, `low` (max-heap of lower half) is either equal in
    *live* size to `high` (min-heap of upper half), or larger by exactly one.
    """

    def __init__(self, window):
        self.window = int(window)
        self.values = deque()       # FIFO of (value, idx) in insertion order
        self.low = []               # max-heap of (-value, idx)
        self.high = []              # min-heap of (value, idx)
        self.in_low = set()         # live indices in `low`
        self.in_high = set()        # live indices in `high`
        self.next_idx = 0

    def _live_low(self):
        return len(self.in_low)

    def _live_high(self):
        return len(self.in_high)

    def _prune_low(self):
        while self.low and self.low[0][1] not in self.in_low:
            heapq.heappop(self.low)

    def _prune_high(self):
        while self.high and self.high[0][1] not in self.in_high:
            heapq.heappop(self.high)

    def _balance(self):
        self._prune_low()
        self._prune_high()
        # low can be at most 1 larger than high
        while self._live_low() > self._live_high() + 1:
            val_neg, idx = self.low[0]
            heapq.heappop(self.low)
            if idx in self.in_low:
                self.in_low.remove(idx)
                heapq.heappush(self.high, (-val_neg, idx))
                self.in_high.add(idx)
            self._prune_low()
        while self._live_high() > self._live_low():
            val, idx = self.high[0]
            heapq.heappop(self.high)
            if idx in self.in_high:
                self.in_high.remove(idx)
                heapq.heappush(self.low, (-val, idx))
                self.in_low.add(idx)
            self._prune_high()

    def push(self, value):
        idx = self.next_idx
        self.next_idx += 1

        # Evict oldest if window is full.
        if len(self.values) == self.window:
            _, old_idx = self.values.popleft()
            if old_idx in self.in_low:
                self.in_low.remove(old_idx)
            elif old_idx in self.in_high:
                self.in_high.remove(old_idx)

        self.values.append((value, idx))
        # Insert: if `low` is empty or value <= current max of low, goes to low; else high.
        self._prune_low()
        if not self.low or value <= -self.low[0][0]:
            heapq.heappush(self.low, (-value, idx))
            self.in_low.add(idx)
        else:
            heapq.heappush(self.high, (value, idx))
            self.in_high.add(idx)
        self._balance()

    def median(self):
        if len(self.values) < self.window:
            return float("nan")
        self._prune_low()
        self._prune_high()
        if self._live_low() == self._live_high():
            return (-self.low[0][0] + self.high[0][0]) / 2.0
        return float(-self.low[0][0])


class EWMASpanNoAdjust:
    """
    Pandas-equivalent `Series.ewm(span=span, adjust=False).mean()`.

    Leading NaNs return NaN; the first non-NaN seeds the recursion; subsequent
    NaNs hold the prior value (mirrors `ewma_mean_span_no_adjust` in utils).
    """

    def __init__(self, span):
        self.alpha = 2.0 / (float(span) + 1.0)
        self.prev = float("nan")
        self.started = False

    def update(self, x):
        if _isnan(x) or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, np.floating) and np.isnan(x)):
            return self.prev if self.started else float("nan")
        if not self.started:
            self.prev = float(x)
            self.started = True
        else:
            self.prev = self.alpha * float(x) + (1.0 - self.alpha) * self.prev
        return self.prev


class EWMSpanStdAdjust:
    """
    Pandas-equivalent `Series.ewm(span=span, adjust=True).std()` (bias-corrected).

    Online weighted accumulators; first valid sample returns NaN
    (n_obs < 2 → undefined); thereafter returns sqrt(unbiased variance).
    Mirrors `ewma_std_span` in utils.
    """

    def __init__(self, span):
        self.alpha = 2.0 / (float(span) + 1.0)
        self.beta = 1.0 - self.alpha
        self.beta_sq = self.beta * self.beta
        self.S_w = 0.0
        self.S_wx = 0.0
        self.S_wxx = 0.0
        self.S_ww = 0.0
        self.n_obs = 0

    def update(self, x):
        if _isnan(x) or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, np.floating) and np.isnan(x)):
            return float("nan")
        v = float(x)
        self.S_w = self.S_w * self.beta + 1.0
        self.S_wx = self.S_wx * self.beta + v
        self.S_wxx = self.S_wxx * self.beta + v * v
        self.S_ww = self.S_ww * self.beta_sq + 1.0
        self.n_obs += 1
        if self.n_obs < 2:
            return float("nan")
        mean = self.S_wx / self.S_w
        var_biased = self.S_wxx / self.S_w - mean * mean
        denom = 1.0 - self.S_ww / (self.S_w * self.S_w)
        if denom <= 0.0:
            return float("nan")
        var_unbiased = var_biased / denom
        if var_unbiased < 0.0:
            var_unbiased = 0.0
        return math.sqrt(var_unbiased)
