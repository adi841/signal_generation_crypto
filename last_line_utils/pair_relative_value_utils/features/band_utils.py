"""
Streaming band construction for QR1_v4. Mirrors the batch expressions in
`get_numba_parameters` (generate_signal_pair_relative_value.py):

    mid        = close.ewm(span=10).mean()                     # adjust=TRUE
    atr14_pct  = clip( wwma(TR,14)/close*100, a14_lo, a14_hi )  # Wilder, IS-frozen clip
    atr50_pct  = clip( wwma(TR,50)/close*100, a50_lo, a50_hi )
    price_n    = atrN_pct * close / 100
    s_upper    = min(mid + nbdev*price14, mid + nbdev*price50)  # pandas skipna min
    s_lower    = max(mid - nbdev*price14, mid - nbdev*price50)  # pandas skipna max
    atr / atr2 = price_n / close * 100    # kernel % units via the price ROUND TRIP
    f_bandwp   = (s_upper - mid) / close * 100

The pct -> price -> pct round trip is deliberate (bitwise parity: skipping it
leaves 1-2 ulp differences on kernel threshold inputs). The clip bounds are the
IS-frozen per-cell constants from data/qr1_cell_constants.parquet — constant for
every bar production will ever see; never recompute them into live data.

The calc also exposes prev_s_middle_last / prev_s_upper_last — the lines as they
were BEFORE the most recent candle update. The kernel's p1 convention reads the
PREVIOUS row's lines on a candle-boundary minute (the fresh candle was not closed
intrabar), and the per-bar live path has no [i-1] to index into.
"""
import math
from collections import deque

from ._primitives import WilderATRPrimitive, EWMASpanAdjustTrue


class QR1BandCalc:

    def __init__(self, nbdev, centre_span, atr_fast_len, atr_slow_len,
                 a14_lo, a14_hi, a50_lo, a50_hi):
        self.nbdev = float(nbdev)
        self._ema = EWMASpanAdjustTrue(int(centre_span))
        self._atr_fast = WilderATRPrimitive(int(atr_fast_len))
        self._atr_slow = WilderATRPrimitive(int(atr_slow_len))
        self.a14_lo = float(a14_lo)
        self.a14_hi = float(a14_hi)
        self.a50_lo = float(a50_lo)
        self.a50_hi = float(a50_hi)

        self.s_middle_last = float("nan")
        self.s_upper_last = float("nan")
        self.s_lower_last = float("nan")
        self.atr_pct_last = float("nan")     # clipped ATR14, kernel % units
        self.atr2_pct_last = float("nan")    # clipped ATR50, kernel % units
        self.f_bandwp_last = float("nan")
        self.prev_s_middle_last = float("nan")
        self.prev_s_upper_last = float("nan")
        self.band_deque = deque(maxlen=3)

    @staticmethod
    def _clip_pct(pct, lo, hi):
        """pandas .clip(lower, upper): NaN passes through untouched (python's
        min/max would silently clamp NaN to a bound — never do that)."""
        if math.isnan(pct):
            return float("nan")
        return min(max(pct, lo), hi)

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        # stash the pre-update lines (the p1-convention "previous candle" lines)
        self.prev_s_middle_last = self.s_middle_last
        self.prev_s_upper_last = self.s_upper_last

        c = float(close)
        mid = self._ema.update(c)

        raw14 = self._atr_fast.update(high, low, c)
        raw50 = self._atr_slow.update(high, low, c)
        pct14 = self._clip_pct(raw14 / c * 100.0, self.a14_lo, self.a14_hi)
        pct50 = self._clip_pct(raw50 / c * 100.0, self.a50_lo, self.a50_hi)
        price14 = pct14 * c / 100.0
        price50 = pct50 * c / 100.0

        # pandas DataFrame.min/max(axis=1) skipna semantics: (NaN, x) -> x
        uf = mid + self.nbdev * price14
        us = mid + self.nbdev * price50
        if math.isnan(uf):
            s_upper = us
        elif math.isnan(us):
            s_upper = uf
        else:
            s_upper = min(uf, us)

        lf = mid - self.nbdev * price14
        ls = mid - self.nbdev * price50
        if math.isnan(lf):
            s_lower = ls
        elif math.isnan(ls):
            s_lower = lf
        else:
            s_lower = max(lf, ls)

        self.s_middle_last = mid
        self.s_upper_last = s_upper
        self.s_lower_last = s_lower
        self.atr_pct_last = price14 / c * 100.0
        self.atr2_pct_last = price50 / c * 100.0
        self.f_bandwp_last = (s_upper - mid) / c * 100.0
        self.band_deque.append((self.s_middle_last, self.s_upper_last, self.s_lower_last))

    @property
    def get_logging_dict(self):
        # Stable latest values (no clear-on-read): log_dump_data runs every minute
        # but _update only fires on candle close.
        return {"s_middle_last": self.s_middle_last,
                "s_upper_last": self.s_upper_last,
                "s_lower_last": self.s_lower_last,
                "atr_pct_last": self.atr_pct_last,
                "atr2_pct_last": self.atr2_pct_last,
                "f_bandwp_last": self.f_bandwp_last}
