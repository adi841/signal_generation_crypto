"""
Streaming primitives for the QR31_v2 pair-lead-lag features.

Each primitive is scalar-state, advances one observation per `update(...)` call,
and matches the pandas semantics of the corresponding batch operation in
`generate_signal_pair_lead_lag.py`. Where SA-directional already ships an
identical primitive (WilderATRPrimitive / EWMASpanNoAdjust), the implementation
is copied verbatim so this sleeve has no cross-module runtime dependency on SA.

The QR31_v2 chain is uniformly adjust=False (EMA8 centre, both z EMAs, Wilder
ATRs, sizing EWMA75) — there is deliberately NO adjust=True EWMA primitive here;
the retired v16_v2 ones (EWMASpanAdjustTrue for the Bollinger calc,
RollingFixedWindowSums for the skew gate, _to_tile_scalar for the vol/sf tiles)
were deleted with that machine.
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
    `ewma_infinite_hist(tr, period, type_='atr')` semantics: the first observation
    seeds the EWMA. TR[0] = high-low (prev_close is NaN → nanmax skips the
    |h-prev_c|/|l-prev_c| terms).
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


class RollingPearsonCorrPrimitive:
    """
    Streaming Pearson correlation over a fixed window N between two series x, y.

    Mirrors the batch path used in `compute_rolling_corr`:
        sx, sy   = bn.move_sum(x, N), bn.move_sum(y, N)
        sxx, syy = bn.move_sum(x², N), bn.move_sum(y², N)
        sxy      = bn.move_sum(x·y, N)
        cov   = sxy - sx·sy / N
        var_x = sxx - sx²  / N
        var_y = syy - sy²  / N
        denom = sqrt(max(var_x · var_y, 0))
        corr  = cov / denom  (NaN where denom == 0)

    NaN-poisoning matches the batch path: any NaN in the window → NaN result
    until N consecutive non-NaN pairs have been observed.
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

