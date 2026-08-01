"""
Streaming primitives for the QR1_v4 pair-relative-value features.

Each primitive is scalar-state, advances one observation per `update(...)` /
`push(...)` call, and matches the semantics of the corresponding pandas
expression in `get_numba_parameters` (generate_signal_pair_relative_value.py).

Copied VERBATIM per the per-sleeve convention (every `features/` package owns its
own primitives, no cross-sleeve runtime dependency):
  - from last_line_utils/pair_lead_lag_utils/features/_primitives.py:
      _isnan, WilderATRPrimitive, EWMASpanNoAdjust, EWMASpanAdjustTrue,
      RollingFixedWindowSums, RollingPearsonCorrPrimitive
  - from last_line_utils/pair_momentum_utils/features/_primitives.py:
      EWMSpanStdAdjust
Momentum's SlidingMedian is not needed here — QR1 has no rolling medians.

Which primitive serves which QR1 feature:
  WilderATRPrimitive          band ATR14 / ATR50 (wwma alpha=1/n on TR)
  EWMASpanAdjustTrue          band centre EMA10 (adjust=True — pandas default)
  EWMASpanNoAdjust            zn's centre EMA10 and outer EMA20 (adjust=False)
  EWMSpanStdAdjust            zn's vol EWM-14 std and EV10 sizing vol (adjust=True)
  RollingFixedWindowSums      skew gate's rolling-50 mean of squared up/down moves
  RollingPearsonCorrPrimitive corr gate's rolling Pearson (DEAD gate, parity only)
"""
import math
from collections import deque

import numpy as np


def _isnan(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, np.floating) and np.isnan(x):
        return True
    return False


class WilderATRPrimitive:
    """
    Wilder ATR (`alpha = 1/period`, adjust=False). Matches
    `tr.ewm(alpha=1/n, adjust=False).mean()` on the batch TR (pd.concat().max(axis=1)
    is skipna, so TR[0] = high-low with the |h-prev_c|/|l-prev_c| terms skipped);
    the first observation seeds the EWMA.
    """

    def __init__(self, period):
        self.alpha = 1.0 / float(period)
        self.atr_curr = float("nan")
        self.prev_close = float("nan")
        self.started = False

    def update(self, high, low, close):
        h, l, c = float(high), float(low), float(close)
        if math.isnan(self.prev_close):
            tr = h - l
        else:
            d1 = abs(h - self.prev_close)
            d2 = abs(l - self.prev_close)
            tr = max(h - l, d1, d2)
        self.prev_close = c
        if not self.started:
            self.atr_curr = tr
            self.started = True
        else:
            self.atr_curr = self.alpha * tr + (1.0 - self.alpha) * self.atr_curr
        return self.atr_curr


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
        if _isnan(x):
            return self.prev if self.started else float("nan")
        if not self.started:
            self.prev = float(x)
            self.started = True
        else:
            self.prev = self.alpha * float(x) + (1.0 - self.alpha) * self.prev
        return self.prev


class EWMASpanAdjustTrue:
    """
    Pandas-equivalent `Series.ewm(span=span, adjust=True, ignore_na=False).mean()`.

    Streaming form of `utils.utils.ewm_mean_adjust_true`: maintains a running
    weighted numerator `num` and denominator `den`. NaN at index i returns NaN
    but still decays the previous history so later weights stay correctly
    normalized (matches the batch implementation exactly).
    """

    def __init__(self, span):
        self.alpha = 2.0 / (float(span) + 1.0)
        self.beta = 1.0 - self.alpha
        self.num = 0.0
        self.den = 0.0
        self.started = False

    def update(self, x):
        if _isnan(x):
            if self.started:
                self.num *= self.beta
                self.den = self.den * self.beta + 1.0
            return float("nan")
        v = float(x)
        if not self.started:
            self.num = v
            self.den = 1.0
            self.started = True
        else:
            self.num = self.num * self.beta + v
            self.den = self.den * self.beta + 1.0
        return self.num / self.den


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
        if _isnan(x):
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


class RollingFixedWindowSums:
    """
    Fixed-window streaming sums over a 1-D series. Maintains a deque of the last
    `window` values plus running sum and sum-of-squares. Yields NaN until the
    window is full (matches `rolling(window)` with the default min_periods).

    NaN values are evicted in the same FIFO order — pushing NaN poisons the
    window: the running sums are marked invalid until `window` consecutive
    non-NaN values have been pushed since the last NaN.
    """

    def __init__(self, window):
        self.window = int(window)
        self.buf = deque(maxlen=self.window)
        self.s = 0.0
        self.ss = 0.0
        self.n_nan_in_window = 0
        self.size = 0  # observations pushed; never exceeds window

    def push(self, x):
        is_nan = _isnan(x)
        if self.size == self.window:
            old = self.buf[0]
            if _isnan(old):
                self.n_nan_in_window -= 1
            else:
                self.s -= old
                self.ss -= old * old
        if is_nan:
            self.buf.append(float("nan"))
            self.n_nan_in_window += 1
        else:
            v = float(x)
            self.buf.append(v)
            self.s += v
            self.ss += v * v
        if self.size < self.window:
            self.size += 1

    def sum(self):
        if self.size < self.window or self.n_nan_in_window > 0:
            return float("nan")
        return self.s

    def sum_sq(self):
        if self.size < self.window or self.n_nan_in_window > 0:
            return float("nan")
        return self.ss

    def mean(self):
        s = self.sum()
        if math.isnan(s):
            return float("nan")
        return s / float(self.window)


class RollingPearsonCorrPrimitive:
    """
    Streaming Pearson correlation over a fixed window N between two series x, y.

        cov   = sxy - sx·sy / N
        var_x = sxx - sx²  / N
        var_y = syy - sy²  / N
        corr  = cov / sqrt(var_x · var_y)   (NaN where the denominator is <= 0)

    NaN-poisoning: any NaN pair in the window → NaN result until N consecutive
    non-NaN pairs have been observed.

    NOTE (QR1): the batch computes `rA.rolling(N).corr(rB)` via pandas' internally
    mean-centred algorithm; this sum-of-products form is algebraically identical
    but can differ at the ~1e-12 level. Acceptable here by design: corr feeds a
    DEAD gate (C_THR = -1 — corr in [-1, 1] can never be below it), so the value
    is kernel-argument parity only, never a decision.
    """

    def __init__(self, window):
        self.window = int(window)
        self.bufx = deque(maxlen=self.window)
        self.bufy = deque(maxlen=self.window)
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0
        self.n_nan_in_window = 0
        self.size = 0

    def push(self, x, y):
        pair_is_nan = _isnan(x) or _isnan(y)
        if self.size == self.window:
            ox = self.bufx[0]
            oy = self.bufy[0]
            old_pair_nan = _isnan(ox) or _isnan(oy)
            if old_pair_nan:
                self.n_nan_in_window -= 1
            else:
                self.sx -= ox
                self.sy -= oy
                self.sxx -= ox * ox
                self.syy -= oy * oy
                self.sxy -= ox * oy
        if pair_is_nan:
            self.bufx.append(float("nan"))
            self.bufy.append(float("nan"))
            self.n_nan_in_window += 1
        else:
            vx = float(x)
            vy = float(y)
            self.bufx.append(vx)
            self.bufy.append(vy)
            self.sx += vx
            self.sy += vy
            self.sxx += vx * vx
            self.syy += vy * vy
            self.sxy += vx * vy
        if self.size < self.window:
            self.size += 1

    def corr(self):
        if self.size < self.window or self.n_nan_in_window > 0:
            return float("nan")
        N = float(self.window)
        cov = self.sxy - self.sx * self.sy / N
        var_x = self.sxx - self.sx * self.sx / N
        var_y = self.syy - self.sy * self.sy / N
        denom_sq = var_x * var_y
        if denom_sq <= 0.0:
            return float("nan")
        return cov / math.sqrt(denom_sq)
