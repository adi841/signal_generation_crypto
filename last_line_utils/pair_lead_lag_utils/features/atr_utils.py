"""
Streaming ATR family for v16_v2 reversal2. Mirrors `compute_atr_pct_clipped` from
`generate_signal_pair_lead_lag.py`:

    tr   = max(h-l, |h-prev_c|, |l-prev_c|)
    atr  = wwma(tr, period) = tr.ewm(alpha=1/period, adjust=False).mean()    [Wilder]
    pct  = atr / close * 100
    pct  = clip(pct, lo_cutoff, hi_cutoff)
    atr  = pct * close / 100                                                  [back to price]

For atr (period=14) and atr2 (period=50) the driver clips with effective +/-inf
(no-op). For atr3 (period=75) the JSON's IS-frozen lo/hi cutoffs apply.

Caller wires three separate instances (one per period). `.atr_price_last` is
the clipped ATR in price units; `.atr_pct_last` is the clipped percent that the
kernel actually consumes for `atr` / `atr2` arguments.
"""
import math
from collections import deque

from ._primitives import WilderATRPrimitive


class ATRPctClippedCalc:
    def __init__(self, period, lo_cutoff=-math.inf, hi_cutoff=math.inf):
        self.period = int(period)
        self.lo_cutoff = float(lo_cutoff)
        self.hi_cutoff = float(hi_cutoff)
        self._atr = WilderATRPrimitive(self.period)
        self.atr_price_deque = deque(maxlen=3)
        self.atr_pct_deque = deque(maxlen=3)
        self.atr_price_last = float("nan")
        self.atr_pct_last = float("nan")
        self._log = {}

    def _initialize(self, df):
        for row in df.itertuples():
            self._update(row.Index, row.open, row.high, row.low, row.close)

    def _update(self, curr_time, open_, high, low, close, **kwargs):
        atr = self._atr.update(high, low, close)
        c = float(close)
        pct = atr / c * 100.0 if c != 0.0 else float("nan")
        if pct < self.lo_cutoff:
            pct = self.lo_cutoff
        elif pct > self.hi_cutoff:
            pct = self.hi_cutoff
        atr_back = pct * c / 100.0
        self.atr_price_last = atr_back
        ## ROUND-TRIP, deliberately: the reference (prep_cell_c) stores the clipped
        ## ATR in PRICE units and converts back to % at the end; the kernel consumes
        ## that round-tripped value, which differs from `pct` by 1 ULP on ~15% of
        ## bars. atr_price_last (the band input) stays the pre-round-trip price.
        self.atr_pct_last = atr_back / c * 100.0 if c != 0.0 else pct
        self.atr_price_deque.append(atr_back)
        self.atr_pct_deque.append(pct)
        self._log = {
            f"atr_p{self.period}_price_last": atr_back,
            f"atr_p{self.period}_pct_last": pct,
        }

    @property
    def get_logging_dict(self):
        # Return the stable latest value every call (no clear-on-read).
        # log_dump_data is called per minute but _update fires on bar
        # completion (60T); returning {} on non-completion minutes makes the
        # per-bucket logging_dict columns inconsistent → pd.DataFrame raises.
        return {
            f"atr_p{self.period}_price_last": self.atr_price_last,
            f"atr_p{self.period}_pct_last":   self.atr_pct_last,
        }
