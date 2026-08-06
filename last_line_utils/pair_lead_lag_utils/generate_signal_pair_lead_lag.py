
import pandas as pd
import numpy as np
import datetime as dt
from functools import lru_cache
import copy
import json
import math
import sys
from pytz import timezone

from config.config_read import config_object

from last_line_utils.pair_lead_lag_utils.pair_lead_lag_utils import PAIR_LEAD_LAG_PARAM_TUPLE, SLEEVE, plain_symbol
from last_line_utils.pair_lead_lag_utils.mult_state import mult_multiplier_array
from utils.pair_lead_lag_utils import cryptopairs_qr31v2_long_iact, TradeOutputPairLeadLagLastOutputCls
from shared_codes.utils.misc_utils import order_tag_generator

import warnings
warnings.filterwarnings("ignore")

#
import os


## ---------------------------------------------------------------------------
## QR31_v2 feature calculations — pandas port of the reference prep
## (crypto_sims/PRODUCTION/engines/pairs_leadlag.py::run_cell +
## engines/core/vendored_pairs.py::prep_cell_c). Deliberately pandas (not
## numpy/bottleneck): this is once-per-init batch code, and identical ops give
## bitwise parity with the golden book.
##
## Unlike B1, the QR31_v2 chain is uniformly adjust=False: the EMA8 centre, both
## z EMAs, every Wilder ATR (ewm alpha=1/n) and the sizing EWMA75 of squared
## log-returns. (prep_cell_c's legacy span-10 adjust=True centre exists in the
## reference but is OVERRIDDEN by the EMA8 rebuild — do not resurrect it.)
##
## The IS-frozen per-cell constants (ATR14/ATR50 clip bounds, sizing scale + clip
## bounds) are READ from a frozen artifact — production init data never reaches the
## 2021-2025 calibration window, so they can never be recomputed here.
##
## NOTE vs B1: entries are MEAN-REVERSION cycles (arm at upper band -> buy at the
## middle -> sell at the ratcheted band), the entry momentum gate is z_ema > 0 via
## msig, the corr EXIT is active (c_thr per TF) while the corr ENTRY gate is disabled
## (entry_c_threshold = -1e18), and sizing carries a cross-pair daily R x I multiplier
## on ENTRY sizing only — deferred until the mult_daily artifact exists (see TODO(mult)
## in get_numba_parameters).
## ---------------------------------------------------------------------------


@lru_cache(maxsize=200)
def load_qr31_cell_constants(coin1, coin2, tf, constants_path):
    """IS-frozen per-cell constants for one (pair, TF) cell, from the pre-dumped
    artifact (built by eod_scripts/pair_lead_lag/build_qr31_cell_constants.py):
    (atr14_lo, atr14_hi, atr50_lo, atr50_hi, sizing_scale, sizing_lo, sizing_hi).

    atr*_lo/hi are the FINAL expanding min/max of the ATR %-of-close series over the
    calibration window; sizing_scale = IS-mean(clipped ATR75 %) / IS-mean(raw EWMA75 %)
    and sizing_lo/hi clip the SCALED sizing-vol series. All constant for every bar
    production will ever process. Raise if the cell is absent: never run unclipped.

    lru_cache is safe — this artifact is rebuilt only if the IS window changed,
    never on a schedule.

    This artifact is the one thing keyed by PLAIN symbols ("BTCUSDT") — the EOD
    builders read `{SYM}PERP-1m-data.parquet`. Live hands us the EXCHANGE-INTERNAL
    names ("BTC-USDT.PERP") that coin_param / submodel_parameters / ohlcv_data use, so
    translate here rather than renaming the symbol everywhere else. plain_symbol is a
    no-op on already-plain input, so the frozen-bundle and offline paths are
    unaffected."""
    coin1, coin2 = plain_symbol(coin1), plain_symbol(coin2)
    df = pd.read_parquet(constants_path)
    row = df[(df['coin1'] == coin1) & (df['coin2'] == coin2) & (df['tf'] == int(tf))]
    if len(row) != 1:
        raise KeyError(f"qr31_cell_constants: no unique row for ({coin1}, {coin2}, {tf}T) "
                       f"in {constants_path} (got {len(row)}). Build all 95 cells with "
                       f"eod_scripts/pair_lead_lag/build_qr31_cell_constants.py")
    r = row.iloc[0]
    return (float(r['atr14_lo']), float(r['atr14_hi']),
            float(r['atr50_lo']), float(r['atr50_hi']),
            float(r['sizing_scale']), float(r['sizing_lo']), float(r['sizing_hi']))


def _wilder_atr_price(df_c):
    """Wilder TR/ATR builder on a candle frame -> dict of RAW price-unit ATRs, keyed by
    period. Verbatim math of the reference ATR (vendored_pairs.py): TR = max(|h-l|,
    |h-pc|, |l-pc|) with pandas skipna (first row = h-l), ATR = TR.ewm(alpha=1/n,
    adjust=False).mean()."""
    tr = pd.concat([(df_c['high'] - df_c['low']).abs(),
                    (df_c['high'] - df_c['close'].shift()).abs(),
                    (df_c['low'] - df_c['close'].shift()).abs()], axis=1).max(axis=1)
    return tr


class GetPairsLeadLagSignal:

    def __init__(self, tup: PAIR_LEAD_LAG_PARAM_TUPLE, df1, df2, comb_df, curr_time, process_name):

        self.tup: PAIR_LEAD_LAG_PARAM_TUPLE = tup
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
        """Build the QR31_v2 kernel inputs for ONE (pair, TF) cell on the minutely
        frame: candle features on the ratio TF frame (EMA8 centre / clipped ATR14+ATR50
        band = centre + nbdev*min / z_ema chain / 4-window leg corr / scaled+clipped
        EWMA75 sizing vol), shifted to the closing minute; per-leg next-2-min OHLC4
        fills; vol-target alloc. Pandas port of the reference
        (PRODUCTION engines/pairs_leadlag.py::run_cell + vendored prep_cell_c).
        Returns [output_tup (kernel-arg order), aux_tup (debug arrays)]."""
        sim_key = self.get_sim_key(tup)
        sc = tup.strategy_config
        tf = int(tup.tf)
        assert config_object.time_zone == "UTC", "QR31_v2 requires UTC day/bin edges"

        cd = self.comb_df

        # ---- minutely working frame (ratio + legs). comb_df is already the outer
        # merge of the legs with ratio O/C + high=max(O,C)/low=min(O,C), ffill+bfill
        # applied column-wise (identical to the reference raw frame after its ffill;
        # legs go missing row-wise so mean/ffill commute for the OHLC4 below).
        mc = pd.DataFrame(index=cd.index)
        mc['mopen'] = cd['open']
        mc['mclose'] = cd['close']
        mc['mhigh'] = cd['high']
        mc['mlow'] = cd['low']
        mc['sc1'] = cd['close_1']
        mc['sc2'] = cd['close_2']
        mc['ob1'] = (cd['open_1'] + cd['high_1'] + cd['low_1'] + cd['close_1']) / 4.0
        mc['ob2'] = (cd['open_2'] + cd['high_2'] + cd['low_2'] + cd['close_2']) / 4.0

        # ---- TF candle frame (reference: dedup/sort + plain resample, KEEP NaN bins) --
        ratio = cd[['open', 'high', 'low', 'close']]
        ratio = ratio.loc[~ratio.index.duplicated(), :].sort_index()
        df = ratio.resample(f"{tf}T").agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        df = df.loc[~df.index.duplicated(), :]

        # ---- IS-frozen per-cell constants (never recomputed) ------------------------
        (atr14_lo, atr14_hi, atr50_lo, atr50_hi,
         sizing_scale, sizing_lo, sizing_hi) = load_qr31_cell_constants(
            tup.coin1, tup.coin2, tf, sc['qr31_cell_constants_path'])

        # ---- TF features (spans/windows off the tuple's strategy_config) -----------
        # Wilder ATRs (RAW, price units) — one TR series, two alphas.
        tr = _wilder_atr_price(df)
        atr14_raw = tr.ewm(alpha=1.0 / int(sc['atr_band_fast_len']), adjust=False).mean()
        atr50_raw = tr.ewm(alpha=1.0 / int(sc['atr_band_slow_len']), adjust=False).mean()

        # % of close, clipped to the IS-frozen bounds. The SAME clipped % series feeds
        # the kernel (second-entry trigger / stop distances) and, in price units, the
        # band construction — the reference computes them twice through two
        # identically-parameterised clip laws (clip_is == clip_atr_pct).
        #
        # ROUND-TRIP, deliberately: prep_cell_c stores the clipped ATR in PRICE units
        # (`role_vol(..) * close / 100`) and converts back to % at the end
        # (`df['atr'] / close * 100`). That float round-trip perturbs ~15% of values
        # by 1 ULP, and the kernel consumes the round-tripped arrays — reproduce it
        # or the batch is not bitwise against the frozen reference.
        atr14_price = (atr14_raw / df['close'] * 100.0).clip(lower=atr14_lo, upper=atr14_hi) \
            * df['close'] / 100.0
        atr50_price = (atr50_raw / df['close'] * 100.0).clip(lower=atr50_lo, upper=atr50_hi) \
            * df['close'] / 100.0
        df['atr14_pct'] = atr14_price / df['close'] * 100.0
        df['atr50_pct'] = atr50_price / df['close'] * 100.0

        # EMA8 centre — ONE representation object: the band centre and the z-score
        # centre are the same EMA (adjust=False), per the frozen v2 spec.
        df['middle'] = df['close'].ewm(span=int(sc['EMA_SPAN']), adjust=False).mean()

        # upper = min(centre + nbdev*ATR14, centre + nbdev*ATR50), price units — built
        # from the PRICE-unit ATRs (pre-round-trip), exactly as the reference's
        # `min(m + nbdev*ap, m + nbdev*a2p)`.
        nbdev = float(tup.nbdev)
        df['upper'] = np.minimum(df['middle'] + nbdev * atr14_price,
                                 df['middle'] + nbdev * atr50_price)

        # z_ema chain (compute_z_ema): (close - EMA8) / RAW Wilder ATR14 (NOT the
        # clipped one), denominator floored at 1e-10, then EMA(span=Z_EMA_LEN,
        # adjust=False). Reference Z_ATR_LEN == atr_band_fast_len == 14 — asserted so a
        # config drift cannot silently reuse the wrong ATR.
        assert int(sc['Z_ATR_LEN']) == int(sc['atr_band_fast_len']), \
            (sc['Z_ATR_LEN'], sc['atr_band_fast_len'])
        z_raw = (df['close'] - df['middle']) / atr14_raw.clip(lower=1e-10)
        df['z_ema'] = z_raw.ewm(span=int(sc['Z_EMA_LEN']), adjust=False).mean()

        # corr: mean over the 4 lookbacks of rolling corr of the two LEGS' candle
        # log-returns; candle-level, reindexed onto the ratio candle frame.
        c1 = cd['close_1'].resample(f"{tf}T").last()
        c2 = cd['close_2'].resample(f"{tf}T").last()
        rA, rB = np.log(c1).diff(), np.log(c2).diff()
        csum = None
        for N in [int(x) for x in sc['corr_lookbacks']]:
            cN = rA.rolling(N).corr(rB)
            csum = cN if csum is None else csum + cN
        df['corr'] = (csum / len(sc['corr_lookbacks'])).reindex(df.index)

        # sizing vol: sqrt(EWMA(span=75, adjust=False) of squared log-returns) * 100,
        # IS-scale-matched to the clipped-ATR75 scale, then IS-clipped (frozen scale
        # and bounds — the expanding pieces of the reference chain live entirely inside
        # the calibration window).
        lr = np.log(df['close']).diff()
        rv_raw = np.sqrt((lr ** 2).ewm(span=int(sc['sizing_vol_ewm_span']), adjust=False).mean()) * 100.0
        df['sizing_vol'] = (rv_raw * sizing_scale).clip(lower=sizing_lo, upper=sizing_hi)

        df = df.rename(columns={'close': 'spc'})

        # ---- merge onto the minutely frame + shift feats to the CLOSING minute -----
        # (reference: outer merge; ROW shift tf-1; p1 from the shifted candle close
        # BEFORE the ffill; then ffill/bfill; forward-2-min OHLC4 fills. The
        # reference's proj() intersects against the ALREADY outer-merged index, so one
        # outer merge of all candle features is equivalent.)
        feats = ['middle', 'upper', 'z_ema', 'corr', 'sizing_vol', 'atr14_pct', 'atr50_pct', 'spc']
        mc = mc.merge(df[feats], left_index=True, right_index=True, how='outer')
        mc[feats] = mc[feats].shift(tf - 1)
        mc['signal_times'] = np.where(~pd.isna(mc['spc']), 1, 0)
        ff = feats + ['mhigh', 'mlow', 'sc1', 'sc2', 'ob1', 'ob2']
        mc[ff] = mc[ff].ffill().bfill()
        mc['next1'] = (mc['ob1'].shift(-1) + mc['ob1'].shift(-2)) / 2.0
        mc['next2'] = (mc['ob2'].shift(-1) + mc['ob2'].shift(-2)) / 2.0

        # ---- trim the fill lookahead: last 2 minutes have NaN next1/next2. The batch
        # never trades them; live processes them incrementally from last_processed_ts.
        mc = mc.iloc[:-2]

        # ---- post-projection transforms (exactly as the reference) ------------------
        z = np.nan_to_num(mc['z_ema'].to_numpy(dtype=np.float64), nan=float(sc['z_projection_fillna']))
        msig = np.where(z > float(sc['entry_z_gate']), 1, -1).astype(np.float64)
        corr = mc['corr'].fillna(float(sc['corr_fillna'])).to_numpy(dtype=np.float64)

        # ---- vol-target alloc + the daily R x I multiplier --------------------------
        # PRODUCTION pairs_leadlag.py::run_cell line 91, verbatim law:
        #     alloc = np.clip(alloc_base * mult.reindex(idx.normalize()).fillna(1.0),
        #                     0.0, 2.0)
        # mult is the cross-market daily sizing state (params.json
        # risk_multiplier_mult; cap MULT_CAP=1.5, floor 0.3) built nightly by
        # eod_scripts/pair_lead_lag/build_mult_daily.py — NO ffill in this law: a
        # missing day sizes at 1.0x (mult_state mirrors that, loudly when stale).
        # This MUST stay in step with the streaming site in
        # pair_lead_lag_last_line_utils.py::update — applying it at only one of the
        # two makes warm-up and live disagree on every entry size.
        alloc_base = pd.Series(
            np.clip((float(sc['VT']) / float(tup.annf)) / mc['sizing_vol'].to_numpy(dtype=np.float64), 0, 1),
            index=mc.index,
        ).bfill().values
        Mm = mult_multiplier_array(sc['mult_daily_path'], mc.index,
                                   logger_name=f"signal_gen_{self.process_name}")
        clip_lo, clip_hi = (float(x) for x in sc['alloc_outer_clip'])
        alloc = np.clip(alloc_base * Mm, clip_lo, clip_hi)

        n = len(mc)
        tp_arr = np.full(n, float(sc['profit_target_tp']))
        override_sig = np.zeros(n, dtype=np.int8)

        ts_array = mc.index.values.astype('datetime64[ns]').astype(np.int64)

        # ---- kernel-input bundle (argument order of cryptopairs_qr31v2_long_iact;
        # the per-cell scalars nbdev/z_thr/x_atr/c_thr ride the tuple, the frozen flags
        # ride strategy_config; `upper` is passed UNMUTATED — generate_signal makes the
        # kernel's mutable copy). ------------------------------------------------------
        output_tup = (
            mc['next1'].to_numpy(dtype=np.float64),          # next_close1
            mc['sc1'].to_numpy(dtype=np.float64),            # same_close1
            tp_arr,                                          # tp
            mc['mhigh'].to_numpy(dtype=np.float64),          # minutely_high
            mc['mlow'].to_numpy(dtype=np.float64),           # minutely_low
            mc['signal_times'].to_numpy(dtype=np.float64),   # p1
            mc['next2'].to_numpy(dtype=np.float64),          # next_close2
            mc['sc2'].to_numpy(dtype=np.float64),            # same_close2
            float(sc['txn_cost']),                           # txn_cost
            msig,                                            # m1
            mc['upper'].to_numpy(dtype=np.float64),          # upper (unmutated)
            mc['middle'].to_numpy(dtype=np.float64),         # middle_line
            mc['atr14_pct'].to_numpy(dtype=np.float64),      # atr (second-entry trigger)
            mc['atr50_pct'].to_numpy(dtype=np.float64),      # atr2 (stop distance)
            float(sc['slippage_per_leg_per_turn']),          # slippage
            alloc,                                           # allocation
            z,                                               # z_ema
            corr,                                            # corr
            override_sig,                                    # override_sig
        )

        aux_tup = (
            ts_array,
            mc['mopen'].to_numpy(dtype=np.float64), mc['mhigh'].to_numpy(dtype=np.float64),
            mc['mlow'].to_numpy(dtype=np.float64), mc['mclose'].to_numpy(dtype=np.float64),
            mc['ob1'].to_numpy(dtype=np.float64), mc['ob2'].to_numpy(dtype=np.float64),
            mc['sizing_vol'].to_numpy(dtype=np.float64),
        )
        return [output_tup, aux_tup]

    def generate_signal(self):
        output_tup, aux_tup = self.get_numba_parameters(self.tup, self.curr_time)

        (ts_array,
         pair_minutely_open, pair_minutely_high, pair_minutely_low, pair_minutely_close,
         coin1_ohlc_based, coin2_ohlc_based,
         sizing_vol) = aux_tup

        (next1, sc1, tp_arr, mhigh, mlow, p1, next2, sc2, txn_cost, msig,
         upper, middle, atr14_pct, atr50_pct, slippage, alloc, z, corr,
         override_sig) = output_tup

        tup = self.tup
        sc = tup.strategy_config

        # Single long-only stream. res2 is identically 0 at the frozen flags (the
        # second entry is VIRTUAL under sec_mode=2 — it only moves the pivot), so
        # there is no second signal stream and no order_tag2.
        # CALLER CONTRACT (see kernel docstring): the kernel MUTATES its band array
        # (pivot ratchet), and upper_line / arm_line / critical_line must be ONE
        # object — pass the same copy twice, exactly as the reference passes `upc`.
        upc = upper.copy()
        array_output, last_output = cryptopairs_qr31v2_long_iact(
            next1, sc1, tp_arr, mhigh, mlow, p1, next2, sc2,
            txn_cost, msig, 1,
            upc, middle, atr14_pct, atr50_pct,
            float(tup.nbdev), slippage, alloc,
            z, float(tup.z_thr), float(tup.x_atr), corr, float(tup.c_thr),
            float(sc['entry_c_threshold']),
            int(sc['arm_always']), int(sc['sec_off']), int(sc['clock_minutes']),
            int(sc['use_override']), override_sig,
            int(sc['sec_delay_min']), int(sc['sec_mode']), int(sc['arm_expiry_min']),
            int(sc['rearm_off']), int(sc['cooldown_min']), upc,
            # z_atr stop selector -- see utils.pair_lead_lag_utils.qr31_z_atr_stop.
            # 0 reproduces production exactly; .get() rather than hard-indexing because
            # live rows in `submodel_parameters` predate these keys.
            int(sc.get('stop_mode', 0)), float(sc.get('stop_pct', 2.0)),
            int(sc.get('signal_invert', 0)),
        )

        # Wrap last-bar state (incl. carried kernel state) for the orchestrator / live.
        numba_output_cls = TradeOutputPairLeadLagLastOutputCls(**last_output._asdict())

        parent_id = self.tup.parent_stratid

        # Trade df: leg fills tradeprice1/tradeprice2 (pair-ratio fill = tp1/tp2,
        # consistent with the reference's log(tp1)-log(tp2) PnL).
        tmp_df = pd.DataFrame({'signal': array_output.res_arr,
                               'tradeprice1': array_output.tradeprice1_arr,
                               'tradeprice2': array_output.tradeprice2_arr}, index=ts_array)
        tmp_df.index = pd.to_datetime(tmp_df.index, unit='ns')
        tmp_df['signal_diff'] = tmp_df['signal'].diff().fillna(0)
        tmp_df['signal_id'] = tmp_df.index.map(lambda x: int(x.timestamp()))
        tmp_df['parent_trading_model'] = parent_id
        tmp_df['is_tp_sl'] = 0

        tmp_df_signal_diff = tmp_df[tmp_df['signal_diff'] != 0]
        tmp_df_signal_diff['price'] = tmp_df_signal_diff['tradeprice1'] / tmp_df_signal_diff['tradeprice2']

        order_tag1 = ""
        order_tag_full = None      # FULL-length stream, for the debug dump only
        if len(tmp_df_signal_diff):
            tmp_df_signal_diff['order_tag'] = tmp_df_signal_diff.apply(
                lambda x: order_tag_generator(ts=x.name, parent_id=int(x['parent_trading_model']), signal=int(x['signal'])),
                axis=1,
            )
            tmp_df.loc[tmp_df_signal_diff.index, ['order_tag', 'price']] = tmp_df_signal_diff[['order_tag', 'price']]
            tmp_df['order_tag'] = tmp_df['order_tag'].ffill()
            tmp_df['price'] = tmp_df['price'].ffill()
            order_tag1 = tmp_df['order_tag'].iloc[-1]  # ffilled "active" tag at last bar

            ## Keep the full-length order-tag stream BEFORE the slice below. create_debug_df
            ## aligns every column against ts_array, so handing it the sliced (last-day)
            ## frame raises a length mismatch (the bug pair_momentum's rewrite fixed).
            order_tag_full = tmp_df['order_tag'].copy()

            ## Slice to the last day window (mirrors SA pattern).
            last_ts = tmp_df.index[-1]
            start_time = last_ts.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(minutes=10)
            tmp_df = tmp_df.loc[start_time:last_ts, :]
            self.dump_pandas_ls.append(tmp_df.copy(deep=True))

        if self.debug:
            self.create_debug_df(
                ts_array,
                # Kernel inputs (per-minute arrays). `upper` is the UNMUTATED band; the
                # kernel's working copy (upc) ends the run ratcheted and is not dumped.
                next_close1=next1, same_close1=sc1, signal_times=p1,
                minutely_high=mhigh, minutely_low=mlow, next_close2=next2, same_close2=sc2,
                middle=middle, upper=upper, atr14_pct=atr14_pct, atr50_pct=atr50_pct,
                z_ema=z, corr=corr, msig=msig, allocation=alloc, sizing_vol=sizing_vol,
                # Pair-ratio + per-leg auxiliary at minute resolution.
                minutely_open=pair_minutely_open, minutely_close=pair_minutely_close,
                coin1_ohlc_based=coin1_ohlc_based, coin2_ohlc_based=coin2_ohlc_based,
                # Kernel array outputs.
                **array_output._asdict(),
                # Order-tag stream, full length (see order_tag_full above).
                order_tag=order_tag_full,
            )

        # Last timestamp the kernel actually processed (= ts_array[-1] AFTER the
        # tail-trim-2 in get_numba_parameters). live.py uses this as the anchor
        # for the first incremental update, so the 2 minutes the batch trim drops
        # are processed incrementally rather than skipped.
        last_processed_ts = pd.Timestamp(int(ts_array[-1]), tz='UTC')

        return numba_output_cls, order_tag1, last_processed_ts

    def create_debug_df(self, ts_array, **kwargs):
        df_ = pd.DataFrame({'ts': ts_array})
        for key, value in kwargs.items():
            if value is None:
                continue
            df_[key] = np.asarray(value)
        df_['ts'] = pd.to_datetime(df_['ts'], unit='ns')

        ## Per-SLEEVE, per-PAIR subdirectory: numpy_pandas_matching is shared by all five
        ## sleeves, and parent_stratids are per-pair LOCAL (every pair reuses
        ## 50000001/03/05/07/09), so a flat path would have two pairs overwrite each
        ## other's dumps. Only self.tup is in scope here — hence the exchange-symbol
        ## form, which also matches the coin1/coin2 columns in qr31_cell_constants.
        out_dir = f"./numpy_pandas_matching/{SLEEVE}/{self.tup.coin1}_{self.tup.coin2}"
        os.makedirs(out_dir, exist_ok=True)
        df_.to_parquet(f"{out_dir}/{self.tup.parent_stratid}_numpy.parquet")
