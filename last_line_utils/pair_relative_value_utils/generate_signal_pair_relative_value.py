
import pandas as pd
import numpy as np
import datetime as dt
from functools import lru_cache
import copy
import json
import math
import os
import sys
from pytz import timezone

from config.config_read import config_object

from last_line_utils.pair_relative_value_utils.pair_relative_value_utils import PAIR_RELATIVE_VALUE_PARAM_TUPLE, plain_symbol
from utils.pair_relative_value_utils import qr1_bandwidth_gate
from utils.pair_relative_value_utils import cryptopairs_qr1v4_short_iact, TradeOutputPairRelativeValueLastOutputCls
from shared_codes.utils.misc_utils import order_tag_generator

import warnings
warnings.filterwarnings("ignore")

#
import os


## ---------------------------------------------------------------------------
## QR1_v4 feature calculations — pandas port of the reference engine
## (crypto_sims/PRODUCTION/engines/pairs_relvalue.py::run_cell +
## engines/core/vendored_qr1.py::prep_cell / minute_score). Deliberately pandas
## (not numpy/bottleneck): this is once-per-init batch code, and the reference
## leans on `ewm(span).std()` adjust=True debiased semantics — identical ops give
## bitwise parity with the golden book. Note the deliberate adjust ASYMMETRY:
## the band-centre EMA10 uses adjust=True (pandas default) while BOTH zn EMAs
## (its EMA10 centre and its EMA20 smoother) use adjust=False; every .std()
## (zn vol, EV10 sizing vol) is adjust=True debiased. Every span/window/tunable
## comes from the frozen snapshot via the tuple / strategy_config — nothing
## hardcoded.
##
## Per-cell IS-frozen constants (kv EV10 sizing scale, wmed wide-band gate
## median, ATR %-clip bounds) are READ from a frozen artifact (production init
## data never reaches the 2021..2025-03 calibration window).
##
## EXPO: the allocation is multiplied by the daily EXPO overlay (market-vol state,
## IS-frozen quantile grid, day D uses closes through D-2). EXPO is a CROSS-MARKET
## daily state a single-pair process cannot derive; it arrives from the shared
## artifact data/expo_daily.parquet, built and refreshed nightly by
## eod_scripts/pair_relative_value/build_expo_daily.py (bitwise-verified against
## the reference build_expo). Dates absent from the artifact resolve to 1.0 — the
## reference's own fillna(1.0) semantics — and the LIVE consumer logs CRITICAL
## when that happens on a current date (a stale artifact must be loud, never
## silent).
## ---------------------------------------------------------------------------


@lru_cache(maxsize=400)
def load_qr1_cell_constants(coin1, coin2, tf, data_dir):
    """IS-frozen per-cell constants for one (pair, TF) cell, from the pre-dumped
    artifact (built once by its own script over the 2021-01-01..2025-03-31
    calibration window; constant for every bar production will ever process):
      kv        EV10 sizing scale = IS-mean(clipped ATR75%) / IS-mean(EV10)
      wmed      IS median of the minute-projected band-width % (wide-band gate)
      atr14_lo/hi, atr50_lo/hi   final expanding-min/max ATR %-clip bounds
    Raise if the cell is absent: never run with unclipped ATRs or an unscaled
    sizing vol.

    This artifact is the one thing keyed by PLAIN symbols ("BTCUSDT") — the EOD
    builders read `{SYM}PERP-1m-data.parquet`. Live hands us the EXCHANGE-INTERNAL
    names ("BTC-USDT.PERP") that coin_param / submodel_parameters / ohlcv_data use, so
    translate here rather than renaming the symbol everywhere else. plain_symbol is a
    no-op on already-plain input, so the frozen-bundle and offline paths are
    unaffected."""
    coin1, coin2 = plain_symbol(coin1), plain_symbol(coin2)
    df = pd.read_parquet(f"{data_dir}/qr1_cell_constants.parquet")
    row = df[(df['coin1'] == coin1) & (df['coin2'] == coin2) & (df['tf'] == int(tf))]
    if len(row) != 1:
        raise KeyError(f"qr1_cell_constants: no unique row for ({coin1}, {coin2}, {tf}T) "
                       f"in {data_dir}/qr1_cell_constants.parquet (got {len(row)})")
    r = row.iloc[0]
    return (float(r['kv']), float(r['wmed']),
            float(r['atr14_lo']), float(r['atr14_hi']),
            float(r['atr50_lo']), float(r['atr50_hi']))


def load_expo_daily(data_dir):
    """Daily EXPO overlay series (naive-UTC date -> expo), already fully lagged (day
    D's row applies to day D and uses closes through D-2). Built by
    eod_scripts/pair_relative_value/build_expo_daily.py, refreshed nightly. Raise if
    the artifact is absent — never run with an unsized overlay. Deliberately NOT
    cached: the live path re-reads it on each UTC date change to pick up the nightly
    rewrite (the batch calls it once per init)."""
    path = f"{data_dir}/expo_daily.parquet"
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise ValueError(f"expo_daily: {path} is empty")
    return pd.Series(df['expo'].to_numpy(), index=pd.DatetimeIndex(df['date']))


@lru_cache(maxsize=8)
def load_entry_score_model(model_path):
    """Frozen additive entry-quality score model (byte-copy of the PRODUCTION
    calibration file shipped in the bundle). The frozen model uses pair-local
    features only — assert that so a re-fitted model needing market-state
    (*_l1) features cannot slip in silently."""
    with open(model_path) as f:
        model = json.load(f)
    ms_feats = [f for f in model['order'] if f.endswith('_l1')]
    assert not ms_feats, f"entry-score model wants market-state features {ms_feats} — unsupported"
    # pre-materialized numpy views for the per-minute scalar scorer (live path):
    model['_np_feats'] = {f: {t: np.array(v, dtype=np.float64) for t, v in model['feats'][f].items()}
                          for f in model['order']}
    model['_np_partials'] = {f: np.array(model['partials'][f], dtype=np.float64)
                             for f in model['order']}
    model['_np_medians'] = {f: {t: float(np.nanmedian(g)) for t, g in model['_np_feats'][f].items()}
                            for f in model['order']}
    return model


def _wilder_atr_pct(df, n):
    """Wilder ATR in % of close on the candle frame — verbatim port of
    vendored_pairs.ATR (tr = max(|h-l|, |h-prev_c|, |l-prev_c|); wwma =
    ewm(alpha=1/n, adjust=False).mean()) with prep_cell's /close*100 scaling."""
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return atr / close * 100.0


def _minute_score(model, tf, p1, mh, ml, middle, upper, f_bandwp):
    """Frozen additive entry-quality score at minute level — verbatim port of
    vendored_qr1.minute_score (decision-time arrays only). Active middle/upper
    follow the kernel's p1 convention (candle-boundary minute uses the previous
    row — the fresh candle was not closed intrabar)."""
    n = len(p1)
    nbins = int(model['nbins'])
    grid = np.linspace(0, 1, 101)

    mid_act = np.where(p1 == 1, np.concatenate(([middle[0]], middle[:-1])), middle)
    up_act = np.where(p1 == 1, np.concatenate(([upper[0]], upper[:-1])), upper)

    # bars since the minutely low last touched the active middle (n if never)
    touch = ml <= mid_act
    idxs = np.arange(n)
    last_touch = np.maximum.accumulate(np.where(touch, idxs, -1))
    tsm = np.where(last_touch >= 0, idxs - last_touch, n).astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        dist_bp = (mh / up_act - 1.0) * 1e4

    vals = {'dist_bp': dist_bp, 'tsm_min': tsm, 'f_bandwp': f_bandwp}
    missing = [f for f in model['order'] if f not in vals]
    assert not missing, f"entry-score model wants unavailable features {missing}"

    score = np.full(n, float(model['intercept'][str(tf)]))
    for f in model['order']:
        q = np.array(model['feats'][f][str(tf)], dtype=np.float64)
        r = np.interp(np.nan_to_num(vals[f], nan=float(np.nanmedian(q))), q, grid)
        b = np.clip((r * nbins).astype(int), 0, nbins - 1)
        score = score + np.array(model['partials'][f], dtype=np.float64)[b]
    return score


def score_entry_quality_scalar(model, tf, f_bandwp, dist_bp, tsm_min):
    """Per-minute SCALAR form of _minute_score for the live path — identical
    grid-interp binning and partial lookup on one observation. Uses the numpy
    views pre-materialized by load_entry_score_model. NaN feature -> the frozen
    grid's median (same as the array path's nan_to_num)."""
    nbins = int(model['nbins'])
    grid = np.linspace(0, 1, 101)
    vals = {'f_bandwp': f_bandwp, 'dist_bp': dist_bp, 'tsm_min': tsm_min}
    score = float(model['intercept'][str(tf)])
    for f in model['order']:
        q = model['_np_feats'][f][str(tf)]
        v = vals[f]
        if v is None or math.isnan(v):
            v = model['_np_medians'][f][str(tf)]
        r = float(np.interp(v, q, grid))
        b = int(r * nbins)
        if b < 0:
            b = 0
        elif b > nbins - 1:
            b = nbins - 1
        score += float(model['_np_partials'][f][b])
    return score


class GetPairsRelativeValueSignal:

    def __init__(self, tup: PAIR_RELATIVE_VALUE_PARAM_TUPLE, df1, df2, comb_df, curr_time, process_name):

        self.tup: PAIR_RELATIVE_VALUE_PARAM_TUPLE = tup
        self.df1 = df1
        self.df2 = df2
        self.comb_df = comb_df
        self.curr_time = curr_time
        self.debug = False
        self.process_name = process_name
        self.dump_pandas_ls = []

    def get_sim_key(self, tup):
        sim_key = "_".join([str(getattr(tup, i)) for i in tup._asdict()])
        print(sim_key)
        return sim_key

    def _update_tup(self, tup):
        self.tup = tup

    # @profile
    def get_numba_parameters(self, tup, curr_time):
        """Build the QR1_v4 kernel inputs for ONE (pair, TF) cell on the minutely
        frame: bands (EMA10 centre + IS-clipped Wilder ATR14/50, min of fast/slow),
        zn, skew / wide-band / (dead) corr gates on the ratio TF candles — shifted
        to the closing minute; per-leg next-2-min OHLC4 fills; EV10 vol-target
        sizing x frozen entry-quality score (EXPO deferred, see module TODO).
        Pandas port of the reference (PRODUCTION engines/pairs_relvalue.py::run_cell
        + vendored_qr1.py::prep_cell).
        Returns [output_tup (kernel-arg order), aux_tup (debug arrays)]."""
        sim_key = self.get_sim_key(tup)
        sc = tup.strategy_config
        tf = int(tup.tf)
        assert config_object.time_zone == "UTC", "QR1_v4 requires UTC day/bin edges"

        cd = self.comb_df

        # ---- minutely working frame (ratio + legs). comb_df is already the outer
        # merge of the legs with ratio O/C + high=max(O,C)/low=min(O,C), ffill+bfill
        # applied column-wise (identical to the reference raw frame after prep's
        # fill; legs go missing row-wise so mean/ffill commute for the OHLC4 below).
        mc = pd.DataFrame(index=cd.index)
        mc['mopen'] = cd['open']
        mc['mclose'] = cd['close']
        mc['mhigh'] = cd['high']
        mc['mlow'] = cd['low']
        mc['sc1'] = cd['close_1']
        mc['sc2'] = cd['close_2']
        mc['ob1'] = (cd['open_1'] + cd['high_1'] + cd['low_1'] + cd['close_1']) / 4.0
        mc['ob2'] = (cd['open_2'] + cd['high_2'] + cd['low_2'] + cd['close_2']) / 4.0

        # ---- TF candle frame (reference: dedup/sort + plain resample, KEEP NaN bins)
        ratio = cd[['open', 'high', 'low', 'close']]
        ratio = ratio.loc[~ratio.index.duplicated(), :].sort_index()
        df = ratio.resample(f"{tf}T").agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        df = df.loc[~df.index.duplicated(), :]

        # ---- frozen per-cell IS constants -------------------------------------------
        kv, wmed, a14_lo, a14_hi, a50_lo, a50_hi = load_qr1_cell_constants(
            tup.coin1, tup.coin2, tf, sc['data_dir'])

        # ---- bands: EMA10 centre (adjust=True) + IS-clipped Wilder ATRs, constr='min'
        nbdev = float(tup.nbdev)
        mid = df['close'].ewm(span=int(sc['band_centre_ema_span'])).mean()
        atr14_pct = _wilder_atr_pct(df, int(sc['atr_band_fast_len'])).clip(lower=a14_lo, upper=a14_hi)
        atr50_pct = _wilder_atr_pct(df, int(sc['atr_band_slow_len'])).clip(lower=a50_lo, upper=a50_hi)
        atr14_price = atr14_pct * df['close'] / 100.0
        atr50_price = atr50_pct * df['close'] / 100.0
        s_upper = pd.concat([mid + nbdev * atr14_price, mid + nbdev * atr50_price], axis=1).min(axis=1)
        s_lower = pd.concat([mid - nbdev * atr14_price, mid - nbdev * atr50_price], axis=1).max(axis=1)

        # ---- zn: the v4 entry AND exit z (engine layer — NOT prep's ATR-based z_ema).
        # adjust=False on BOTH EMAs; the vol leg is price-unit EWM std, adjust=True.
        lr = np.log(df['close']).diff()
        zn = ((df['close'] - df['close'].ewm(span=int(sc['zn_centre_ema_span']), adjust=False).mean())
              / (lr.ewm(span=int(sc['zn_vol_ewm_span'])).std() * df['close']
                 ).clip(lower=float(sc['zn_denom_floor']))
              ).ewm(span=int(sc['zn_smooth_ewm_span']), adjust=False).mean()

        # ---- per-LEG candle features (skew gate + dead corr) — on the RAW leg closes
        # (self.df1/df2 are unfilled; the reference resamples the unfilled raw legs, so
        # a whole-bin data gap must yield NaN, not a carried close).
        cA = self.df1['close'].resample(f"{tf}T").last()
        cB = self.df2['close'].resample(f"{tf}T").last()
        rA, rB = np.log(cA).diff(), np.log(cB).diff()
        csum = None
        for N in [int(x) for x in sc['corr_lookbacks']]:
            cN = rA.rolling(N).corr(rB)
            csum = cN if csum is None else csum + cN
        corr_series = (csum / len(sc['corr_lookbacks'])).reindex(df.index)

        skN = int(sc['SKEW_N'])

        def _skew_score(close, N):
            r = np.log(close).diff()
            up = r.where(r > 0, 0.0)
            dn = (-r).where(r < 0, 0.0)
            upv = np.sqrt((up ** 2).rolling(N).mean()).clip(lower=1e-10)
            dnv = np.sqrt((dn ** 2).rolling(N).mean()).clip(lower=1e-10)
            return np.log(upv / dnv)

        # gate TRUE when skew(leg2) > skew(leg1), shifted 1 candle (adverse-skew exclusion)
        sk_diff = (_skew_score(cB, skN) - _skew_score(cA, skN)).shift(1)
        sk_gate = (sk_diff > 0).astype(np.float64).reindex(df.index)

        # ---- sizing vol + band width (candle level) ---------------------------------
        ev10 = lr.ewm(span=int(sc['sizing_vol_ewm_span'])).std() * 100.0
        f_bandwp = (s_upper - mid) / df['close'] * 100.0

        # ---- merge onto the minutely frame + shift feats to the CLOSING minute ------
        # (reference: outer merge; ROW shift tf-1; p1 from the shifted candle close
        # BEFORE the ffill; then ffill/bfill; forward-2-min OHLC4 fills.)
        # atr/atr2 go to the kernel in % units via the reference's price-unit round
        # trip (pct -> price for the bands -> back to pct) — bitwise-parity detail:
        # skipping the round trip leaves 1-2 ulp differences on threshold inputs.
        proj = pd.DataFrame({'s_middle': mid, 's_upper': s_upper, 's_lower': s_lower,
                             'zn': zn, 'corr': corr_series, 'sk_gate': sk_gate,
                             'spc': df['close'],
                             'atr': atr14_price / df['close'] * 100.0,
                             'atr2': atr50_price / df['close'] * 100.0,
                             'f_bandwp': f_bandwp, 'ev10': ev10})
        feats = list(proj.columns)
        mc = mc.merge(proj, left_index=True, right_index=True, how='outer')
        mc[feats] = mc[feats].shift(tf - 1)
        mc['signal_times'] = np.where(~pd.isna(mc['spc']), 1, 0)
        ff = feats + ['mhigh', 'mlow', 'sc1', 'sc2', 'ob1', 'ob2']
        mc[ff] = mc[ff].ffill().bfill()
        mc['next1'] = (mc['ob1'].shift(-1) + mc['ob1'].shift(-2)) / 2.0
        mc['next2'] = (mc['ob2'].shift(-1) + mc['ob2'].shift(-2)) / 2.0

        # ---- trim the fill lookahead: last 2 minutes have NaN next1/next2. The batch
        # never trades them; live processes them incrementally from last_processed_ts.
        # (Config's FILL_WINDOW_MIN says 1 minute for QR1_v4, but the frozen engine and
        # the goldens use the 2-minute mean — the code wins; positions are unaffected.)
        mc = mc.iloc[:-2]

        # ---- decision-time arrays ----------------------------------------------------
        p1 = mc['signal_times'].to_numpy(dtype=np.float64)
        mh = mc['mhigh'].to_numpy(dtype=np.float64)
        ml = mc['mlow'].to_numpy(dtype=np.float64)
        middle = mc['s_middle'].to_numpy(dtype=np.float64)
        upper = mc['s_upper'].to_numpy(dtype=np.float64)
        zn_m = mc['zn'].fillna(float(sc['zn_projection_fillna'])).to_numpy(dtype=np.float64)
        skew_ok = (mc['sk_gate'].fillna(0.0).to_numpy() >= 0.5)
        fbw = mc['f_bandwp'].to_numpy(dtype=np.float64)

        # ---- entry signal: z-gate AND band-width gate AND skew gate -------------------
        # The band-width conjunct is selectable (see utils.pair_relative_value_utils.
        # qr1_bandwidth_gate); gate_mode 0 reproduces `fbw > wmed` exactly. The SAME
        # helper drives the live per-minute path, so batch and live cannot diverge.
        bw_gate = qr1_bandwidth_gate(
            fbw, wmed,
            mc['atr'].to_numpy(dtype=np.float64), mc['atr2'].to_numpy(dtype=np.float64),
            mc['corr'].fillna(1.0).to_numpy(dtype=np.float64), float(sc['C_THR']),
            int(sc.get('bandwidth_gate_mode', 0)))
        msig = np.where((zn_m > float(tup.z_entry)) & skew_ok & bw_gate, 1, -1
                        ).astype(np.float64)

        # ---- sizing: EV10 fast-vol allocation, IS-mean matched via kv; entry-quality
        # score (frozen additive model), two-stage clip around the IS median.
        # NOTE (candle-fresh): via the shift(tf-1) projection, a candle's EV10 lands on
        # that candle's own closing minute — the kernel reads allocation[i] directly at
        # the entry minute, so entries decided on a boundary minute are sized off the
        # just-closed candle. The live path must reproduce exactly this freshness.
        ev10_m = mc['ev10'].to_numpy(dtype=np.float64)
        al_ew = pd.Series(
            np.clip((float(sc['VT']) / (ev10_m * kv).clip(1e-9)) / float(tup.annf), 0, 1),
            index=mc.index,
        ).bfill().values

        model = load_entry_score_model(sc['entry_score_model_path'])
        score = _minute_score(model, tf, p1, mh, ml, middle, upper, fbw)
        # Live-seed stash: minutes-since-middle-touch at the LAST processed bar.
        # Consumed by the strat obj's initialize() right after generate_signal()
        # so the live tsm counter (the score's tsm_min feature) continues
        # seamlessly across the batch->live handoff. None = never touched.
        mid_act_ = np.where(p1 == 1, np.concatenate(([middle[0]], middle[:-1])), middle)
        lt_ = np.maximum.accumulate(np.where(ml <= mid_act_, np.arange(len(p1)), -1))
        self._tsm_seed = int(len(p1) - 1 - lt_[-1]) if lt_[-1] >= 0 else None
        sc_lo, sc_hi = [float(x) for x in sc['SC_CLIP']]
        med = float(model['med_entry_score'])
        # Daily EXPO overlay (reference: expo_d.reindex(idx.normalize()).fillna(1.0));
        # the artifact index is naive UTC, the minute index may be tz-aware live.
        expo_s = load_expo_daily(sc['data_dir'])
        _dates = mc.index.normalize()
        if _dates.tz is not None:
            _dates = _dates.tz_localize(None)
        expo_arr = expo_s.reindex(_dates).fillna(1.0).to_numpy()
        # multiplication order is the reference's: (al_ew * clip) * expo — bitwise.
        alloc = al_ew * np.clip(score / med, sc_lo, sc_hi) * expo_arr

        ts_array = mc.index.values.astype('datetime64[ns]').astype(np.int64)

        # ---- kernel-input bundle (per-bar argument order of cryptopairs_qr1v4_short_iact;
        # the per-cell scalars z_thr/x_atr/C_THR/z_entry/SEC_MULT and the constants
        # m2=1 / tp=1000.0 ride the tuple at the call site). ---------------------------
        output_tup = (
            mc['next1'].to_numpy(dtype=np.float64),        # next_close1
            mc['sc1'].to_numpy(dtype=np.float64),          # same_close1
            p1,                                            # p1
            mh,                                            # minutely_high
            ml,                                            # minutely_low
            mc['next2'].to_numpy(dtype=np.float64),        # next_close2
            mc['sc2'].to_numpy(dtype=np.float64),          # same_close2
            msig,                                          # m1 (gated entry signal)
            upper,                                         # upper_line
            middle,                                        # middle_line
            mc['s_lower'].to_numpy(dtype=np.float64),      # lower_line (parity; unused by short kernel)
            mc['atr'].to_numpy(dtype=np.float64),          # atr  (clipped ATR14 %)
            mc['atr2'].to_numpy(dtype=np.float64),         # atr2 (clipped ATR50 %)
            np.ascontiguousarray(alloc),                   # allocation
            np.ascontiguousarray(zn_m),                    # z (zn — entry AND exit z)
            mc['corr'].fillna(1.0).to_numpy(dtype=np.float64),  # corr (dead gate)
            skew_ok,                                       # skew gate bool
            float(sc['txn_cost']),                         # tc
            float(sc['slippage_per_leg_per_turn']),        # slip
        )

        aux_tup = (
            ts_array,
            mc['mopen'].to_numpy(dtype=np.float64), mc['mhigh'].to_numpy(dtype=np.float64),
            mc['mlow'].to_numpy(dtype=np.float64), mc['mclose'].to_numpy(dtype=np.float64),
            mc['ob1'].to_numpy(dtype=np.float64), mc['ob2'].to_numpy(dtype=np.float64),
            ev10_m, np.ascontiguousarray(score), np.ascontiguousarray(al_ew),
        )
        return [output_tup, aux_tup]

    def generate_signal(self):
        output_tup, aux_tup = self.get_numba_parameters(self.tup, self.curr_time)

        (ts_array,
         pair_minutely_open, pair_minutely_high, pair_minutely_low, pair_minutely_close,
         coin1_ohlc_based, coin2_ohlc_based,
         ev10, score, al_ew) = aux_tup

        (next1, sc1, p1, mh, ml, next2, sc2, msig, upper, middle, lower,
         atr, atr2, alloc, zn, corr, skew_ok, txn_cost, slippage) = output_tup

        tup = self.tup
        sc = tup.strategy_config

        # DUAL-tranche short stream — one kernel, res (entry 1) + res2 (scale-in),
        # each half the cell allocation. tp is the vestigial profit-target input
        # (constant 1000.0 in the frozen reference); m2=1 selects the msig=+1 regime.
        n = next1.shape[0]
        tp = np.full(n, 1000.0)
        array_output, last_output = cryptopairs_qr1v4_short_iact(
            next1, sc1, tp, mh, ml, p1,
            next2, sc2,
            txn_cost, msig, 1,
            upper.copy(), middle.copy(), lower.copy(),
            atr, atr2, float(sc['SEC_MULT']),
            slippage, alloc,
            zn, float(tup.z_thr), float(tup.x_atr),
            corr, float(sc['C_THR']),
            float(tup.z_entry), skew_ok,
            # direction inversion -- see utils.pair_relative_value_utils kernel comment
            int(sc.get('signal_invert', 0)),
        )

        # Wrap last-bar state (incl. carried kernel state) for the orchestrator / live.
        numba_output_cls = TradeOutputPairRelativeValueLastOutputCls(**last_output._asdict())

        parent_id = self.tup.parent_stratid

        # ---- two order-tag streams: tranche 1 (res, fills tp1/tp2) and the scale-in
        # tranche 2 (res2, fills tp3/tp4) — the reversal2/ancestor dual-stream contract
        # the live layer consumes as order_tag1/order_tag2. Both streams are keyed to
        # THIS cell's parent id here; how tranche-2 rows are keyed downstream (ancestor
        # live.py posts them at parent_id + 2) is decided in the live step.
        def _tag_stream(res_arr, tp_a, tp_b, stream_parent_id):
            ## `stream_parent_id` is the id the DB ROWS are keyed to; `parent_id` is the
            ## id the ORDER TAG is keyed to. They differ for the scale-in tranche, and
            ## the split is deliberate — it mirrors live.py exactly, which posts tranche 2
            ## at `parent_id + 2` (populate_db_signal_dict(second_entry_parent_id, ...))
            ## while still generating its tag with the BASE id
            ## (`get_order_tag(curr_time, parent_id, res2)`). Keying the rows to
            ## `parent_id` for both streams — as this did before — made the tranche-2
            ## rows collide with tranche 1 on the live_signals PK and silently overwrite
            ## them through the ON CONFLICT DO UPDATE.
            tmp_df = pd.DataFrame({'signal': res_arr,
                                   'tradeprice1': tp_a,
                                   'tradeprice2': tp_b}, index=ts_array)
            tmp_df.index = pd.to_datetime(tmp_df.index, unit='ns')
            tmp_df['signal_diff'] = tmp_df['signal'].diff().fillna(0)
            tmp_df['signal_id'] = tmp_df.index.map(lambda x: int(x.timestamp()))
            tmp_df['parent_trading_model'] = stream_parent_id
            tmp_df['is_tp_sl'] = 0
            ## live_signals columns that are NOT NULL but carry no QR1 meaning. Both are
            ## required by Base.dump_hist_signal_df, which reads them off this frame by
            ## attribute — without them the live warm-up backfill dies with
            ## AttributeError before the per-minute loop ever starts.
            ##   case_num       QR1 has no case machinery; 0 is the sleeve's live value
            ##                  (live.py's per-minute DB rows carry the same constant).
            ##   execution_type the broker execution style, off the DB param row.
            tmp_df['case_num'] = 0
            tmp_df['execution_type'] = self.tup.exec_type

            tmp_df_signal_diff = tmp_df[tmp_df['signal_diff'] != 0]
            tmp_df_signal_diff['price'] = tmp_df_signal_diff['tradeprice1'] / tmp_df_signal_diff['tradeprice2']

            order_tag = ""
            if len(tmp_df_signal_diff):
                ## The tag is keyed to the BASE cell id, NOT to stream_parent_id — do not
                ## "simplify" this back to reading x['parent_trading_model']. live.py
                ## builds BOTH tranches' tags off the base id
                ## (`get_order_tag(curr_time, parent_id, res2)` for tranche 2), so keying
                ## the backfill to parent_id + 2 would hand the live process a seed tag it
                ## would then contradict on its very next transition.
                tmp_df_signal_diff['order_tag'] = tmp_df_signal_diff.apply(
                    lambda x: order_tag_generator(ts=x.name, parent_id=parent_id,
                                                  signal=int(x['signal'])),
                    axis=1,
                )
                tmp_df.loc[tmp_df_signal_diff.index, ['order_tag', 'price']] = tmp_df_signal_diff[['order_tag', 'price']]
                tmp_df['order_tag'] = tmp_df['order_tag'].ffill()
                tmp_df['price'] = tmp_df['price'].ffill()
                ## Bars BEFORE the cell's first-ever transition have no tag to carry.
                ## live_signals.order_tag is NOT NULL, and a leftover float nan reaches
                ## postgres as the literal string 'NaN', which reads like a real tag —
                ## so use the same "" sentinel this function already returns.
                tmp_df['order_tag'] = tmp_df['order_tag'].fillna("")
                order_tag = tmp_df['order_tag'].iloc[-1]  # ffilled "active" tag at last bar

                ## Slice to the last day window (mirrors SA pattern).
                last_ts = tmp_df.index[-1]
                start_time = last_ts.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(minutes=10)
                tmp_df = tmp_df.loc[start_time:last_ts, :]
                self.dump_pandas_ls.append(tmp_df.copy(deep=True))
            return order_tag

        order_tag1 = _tag_stream(array_output.res_arr,
                                 array_output.tradeprice1_arr, array_output.tradeprice2_arr,
                                 stream_parent_id=parent_id)
        order_tag2 = _tag_stream(array_output.res2_arr,
                                 array_output.tradeprice3_arr, array_output.tradeprice4_arr,
                                 stream_parent_id=parent_id + 2)

        if self.debug:
            self.create_debug_df(
                ts_array,
                # Kernel inputs (per-minute arrays).
                next_close1=next1, same_close1=sc1, signal_times=p1,
                minutely_high=mh, minutely_low=ml, next_close2=next2, same_close2=sc2,
                msig=msig, upper=upper, middle=middle, lower=lower,
                atr=atr, atr2=atr2, allocation=alloc, zn=zn, corr=corr,
                skew_ok=skew_ok.astype(np.float64),
                ev10=ev10, entry_score=score, alloc_ew=al_ew,
                # Pair-ratio + per-leg auxiliary at minute resolution.
                minutely_open=pair_minutely_open, minutely_close=pair_minutely_close,
                coin1_ohlc_based=coin1_ohlc_based, coin2_ohlc_based=coin2_ohlc_based,
                # Kernel array outputs (both tranches).
                **array_output._asdict(),
            )

        # Last timestamp the kernel actually processed (= ts_array[-1] AFTER the
        # tail-trim-2 in get_numba_parameters). live.py uses this as the anchor
        # for the first incremental update, so the 2 minutes the batch trim drops
        # are processed incrementally rather than skipped.
        last_processed_ts = pd.Timestamp(int(ts_array[-1]), tz='UTC')

        return numba_output_cls, order_tag1, order_tag2, last_processed_ts

    def create_debug_df(self, ts_array, **kwargs):
        df_ = pd.DataFrame({'ts': ts_array})
        for key, value in kwargs.items():
            if value is None:
                continue
            df_[key] = np.asarray(value)
        df_['ts'] = pd.to_datetime(df_['ts'], unit='ns')
        ## Per-sleeve/per-pair subdir so concurrent hist_replay dumps of different
        ## sleeves/pairs never collide (momentum's pattern + the sleeve segment).
        out_dir = f"./numpy_pandas_matching/pair_relative_value/{self.tup.coin1}_{self.tup.coin2}"
        os.makedirs(out_dir, exist_ok=True)
        df_.to_parquet(f"{out_dir}/{self.tup.parent_stratid}_numpy.parquet")
