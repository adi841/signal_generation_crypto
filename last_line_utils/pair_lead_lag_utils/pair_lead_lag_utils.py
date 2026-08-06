
import json
import math
import os
from collections import namedtuple

## QR31_v2 per-CELL parameters. ONE instance == ONE (pair, TF) sleeve — the
## granularity at which the research produces an independent daily-return stream and
## the live framework runs one aggregator + one kernel -> one signal. 5 TFs per pair.
## Long-only MEAN-REVERSION cycles on the leg1/leg2 ratio: arm when the ratio touches
## the upper band, enter long on the pullback to the EMA8 middle, exit at the
## (pivot-ratcheted) band or on a z/ATR or corr stop. ONE real tranche per trade —
## the "second entry" is VIRTUAL (sec_mode=2): it books no position and no cost, it
## only moves the pivot that anchors the stop/target. Entry sizing carries a cross-pair
## daily R x I multiplier (params.json risk_multiplier_mult), read from the nightly
## artifact via strategy_config['mult_daily_path'].
## Per-TF tunables (nbdev/z_thr/x_atr/c_thr/annf) ride the tuple top-level (kernel
## scalars); shared globals ride strategy_config — everything is sourced from
## pair_lead_lag_production/params.json, nothing hardcoded in code.
PAIR_LEAD_LAG_PARAM_TUPLE = namedtuple("pair_lead_lag_param_tuple", [
    "parent_stratid",
    "coin1", "coin2",       # leg1 (major) / leg2 — ratio = coin1/coin2
    "tf",                   # timeframe minutes (15/30/60/120/240)
    "agg_time",             # f"{tf}T" (OHLCV aggregator format)
    "nbdev",                # band width (upper = centre + nbdev*min(ATR14, ATR50));
                            # ALSO the virtual-second-entry trigger distance (nbdev*atr14_pct)
    "z_thr",                # stop momentum gate: z/ATR stop fires only while z_ema < z_thr
    "x_atr",                # stop distance BELOW THE PIVOT, in clipped-ATR50 % units
    "c_thr",                # corr EXIT threshold: exit + disarm when corr < c_thr (ACTIVE here)
    "annf",                 # sqrt(PER*24*60/tf) — derived exact, asserted vs params.json
    "strategy_config",      # nested dict: globals + costs + artifact paths
])

## Per-bar kernel inputs assembled by the live last-line object each minute (features
## refresh on tf-bar close). QR31_v2-shaped: only the RATIO-side quantities plus costs —
## the four per-leg price arguments the kernel also takes (next_close1/2, same_close1/2)
## are deliberately ABSENT: live.py owns the legs and passes them straight to the kernel
## from its INPUT_DATA_TUPLE. The per-cell scalars nbdev/z_thr/x_atr/c_thr and the
## frozen flags come off PAIR_LEAD_LAG_PARAM_TUPLE / strategy_config at the call site.
##
## THE *_prev FIELDS: on candle-close minutes (p1==1) the frozen kernel reads the
## PREVIOUS minute's line values ([i-1] convention) for the entry middle test and the
## virtual-second trigger's atr — i.e. the candle-BEFORE-this-close values. On non-p1
## minutes prev == current. `upper_eff_prev` (the post-ratchet band the arm/exit/re-arm
## [i-1] reads see) is deliberately NOT here — the ratchet is TRADE state, carried in
## numba_cls.upper_eff_prev_lo, not a feature. z_ema / corr / atr50 / allocation are
## [i]-read by the kernel -> current values only.
PAIR_LEAD_LAG_NUMBA_INPUT_TUPLE = namedtuple('PairLeadLagNumbaInputTup', [
    'p1',               # 1 when a tf-bar just closed, else 0
    'minutely_high',    # ratio minutely high = max(open, close) (arm + band exit + triggers)
    'minutely_low',     # ratio minutely low = min(open, close) (entry + virtual trigger + stop)
    'middle',           # EMA(EMA_SPAN=8, adjust=False) of ratio candle closes
    'middle_prev',      # previous minute's middle ([i-1] read on p1 minutes)
    'upper',            # RAW band: min(middle + nbdev*ATR14, middle + nbdev*ATR50), price units
    'atr14_pct',        # clipped Wilder ATR14 % of close (virtual-second trigger distance)
    'atr14_pct_prev',   # previous minute's atr14_pct ([i-1] read on p1 minutes)
    'atr50_pct',        # clipped Wilder ATR50 % of close (z/ATR stop distance)
    'z_ema',            # z chain (EMA8 centre / raw ATR14 / EMA20 smooth), NaN -> -1.0
    'm1',               # msig: +1.0 iff z_ema > entry_z_gate, else -1.0 (entry momentum gate)
    'corr',             # mean 4-window rolling corr of LEG candle log-returns, NaN -> 1.0
    'txn_cost',         # FIXED_COST_per_trade (frozen-bundle value; live passes runtime config)
    'slippage',         # per-pair slip = mean of the two legs' per-symbol slippage
    'allocation',       # clip( clip((VT/annf)/sizing_vol, 0, 1) * mult_daily, 0, 2 )
])


## Sleeve name — namespaces the shared output directories (entry_data_dir/,
## numpy_pandas_matching/), which all five sleeves write into with per-pair-LOCAL
## parent ids (every pair reuses 50000001/03/05/07/09 here).
SLEEVE = "pair_lead_lag"

## Authoritative frozen production bundle for THIS sleeve. Named by sleeve, not by
## research version — the version it pins is declared inside the file ("strategy":
## "QR31_v2"). Mirrors crypto_sims/PRODUCTION (config.py QR31 block +
## engines/pairs_leadlag.py + engines/core/vendored_pairs.py QR31_CFG); every value in
## it is machine-verified against that source (verify_pll_params.py, 104 checks).
PAIR_LEAD_LAG_PARAMS_PATH = "/home/rocky/crypto_sims/pair_lead_lag_production/params.json"

## Where the EOD builders write. Resolved from this file's location rather than hardcoded,
## so a checkout on another host works without editing paths.
EOD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eod_scripts", "pair_lead_lag", "output")

## parent_stratid band map: 1xxxxxxx directional_momentum, 2xxxxxxx retired (dead
## B1_v8_v9 rows at 20000001..20000006), 3xxxxxxx pair_relative_value (previously legacy
## hjson pair_reversal — those DB rows are dead), 4xxxxxxx pair_momentum (previously
## legacy pair_breakout — dead), 5xxxxxxx pair_lead_lag, 6/7xxxxxxx legacy single-asset,
## 8/9xxxxxxx free. This sleeve MOVED OFF its historic 2xxxxxxx band: the retired
## B1_v8_v9_LONG incarnation's DB rows and entry_data_dir CSVs there are dead; never
## read them (nor any old-coding 5xxxxxxx pair_momentum artifacts) as QR31_v2 output.
## IDs are per-pair LOCAL (every pair reuses 50000001/03/05/07/09) — global uniqueness
## is applied downstream when registering child_id_info.
## STRIDE 2: the house convention for single-entry PAIR strategies reserves 2 ids per
## (pair, TF) cell — the signal stream posts at +0 and the +1 slot is the leg-2 id
## used downstream at child-id registration. (2-entry pair strategies reserve 4.)
PAIR_LEAD_LAG_START_PARENT_ID = 50000001
PAIR_LEAD_LAG_PARENT_ID_STRIDE = 2


def build_pair_lead_lag_parameter_dict(coin1, coin2, params_path=PAIR_LEAD_LAG_PARAMS_PATH,
                                       start_parent_id=PAIR_LEAD_LAG_START_PARENT_ID):
    """Build {parent_stratid: PAIR_LEAD_LAG_PARAM_TUPLE} for ONE pair — all 5 TF cells.

    One tuple == one (pair, TF) sleeve; parent_stratids are per-pair local, assigned
    `start_parent_id + 2*tf_index` (stride 2 — single-entry pair convention: the +1
    slot is the leg-2 id; QR31's second entry is virtual, so there is no second
    stream to number). Everything is read from params.json: per-TF
    nbdev/z_thr/x_atr/c_thr, global
    constants, per-symbol slippage (pair slip = mean of legs). annf is derived exact
    from annualization_days_PER (params.json stores 2dp-rounded values) and asserted
    against the json. Pure / side-effect free so the matching harness can use it
    without the live Base/socket machinery.

    The two EOD-built artifacts are surfaced as explicit paths (`qr31_cell_constants_path`,
    `mult_daily_path`) following the pair_momentum convention; their builders live in
    eod_scripts/pair_lead_lag/ (NOT YET BUILT — loaders must fail loudly on absence).
    The shipped frozen activity-counts history (input to the mult builder, not to the
    cells) is exposed as `activity_counts_path` off the bundle's data_dir.
    """
    with open(params_path) as f:
        payload = json.load(f)

    g = payload["global_constants"]
    slip_map = payload["slippage_per_symbol_fraction"]
    pairs = [tuple(p) for p in payload["universe"]["pairs_leg1_leg2"]]
    if (coin1, coin2) not in pairs:
        raise KeyError(f"({coin1}, {coin2}) not in the QR31_v2 universe ({len(pairs)} pairs). "
                       f"Note QR31 is the WIDEST pairs universe (BNB and DOT/UNI all in) — "
                       f"it is not the B1 or QR1 universe.")

    slippage = (float(slip_map[coin1]) + float(slip_map[coin2])) / 2.0
    data_dir = f"{os.path.dirname(params_path)}/data"

    strategy_config = {
        "asset": coin1,
        "pair_asset": coin2,
        "long_only": 1,
        "VT": float(g["VT_vol_target"]),
        "txn_cost": float(g["FIXED_COST_per_trade"]),
        "slippage_per_leg_per_turn": slippage,
        "EMA_SPAN": int(g["EMA_SPAN"]),
        "Z_ATR_LEN": int(g["Z_ATR_LEN"]),
        "Z_EMA_LEN": int(g["Z_EMA_LEN"]),
        "entry_z_gate": float(g["entry_z_gate"]),
        "z_projection_fillna": float(g["z_projection_fillna"]),
        "corr_lookbacks": list(g["corr_lookbacks"]),
        "corr_fillna": float(g["corr_fillna"]),
        "atr_band_fast_len": int(g["atr_band_fast_len"]),
        "atr_band_slow_len": int(g["atr_band_slow_len"]),
        "sizing_vol_ewm_span": int(g["sizing_vol_ewm_span"]),
        "MULT_CAP": float(g["MULT_CAP"]),
        "alloc_outer_clip": list(g["alloc_outer_clip"]),
        "sec_mode": int(g["sec_mode"]),
        "sec_delay_min": int(g["sec_delay_min"]),
        "cooldown_min": int(g["cooldown_min"]),
        "arm_expiry_min": int(g["arm_expiry_min"]),
        "arm_always": int(g["arm_always"]),
        "sec_off": int(g["sec_off"]),
        "clock_minutes": int(g["clock_minutes"]),
        "use_override": int(g["use_override"]),
        "rearm_off": int(g["rearm_off"]),
        "entry_c_threshold": float(g["entry_c_threshold"]),
        "profit_target_tp": float(g["profit_target_tp"]),
        "calibration_window": list(g["calibration_window"]),
        "CALIBRATION_END": str(g["CALIBRATION_END"]),
        "annualization_days_PER": int(g["annualization_days_PER"]),
        "params_path": params_path,
        "data_dir": data_dir,
        "activity_counts_path": f"{data_dir}/leadlag_activity_counts.parquet",
        "qr31_cell_constants_path": f"{EOD_OUTPUT_DIR}/qr31_cell_constants.parquet",
        "mult_daily_path": f"{EOD_OUTPUT_DIR}/mult_daily.parquet",
        ## z_atr stop selector -- see utils.pair_lead_lag_utils.qr31_z_atr_stop.
        ## 0 reproduces production bit-for-bit. .get(): frozen bundles predate these keys.
        "stop_mode": int(g.get("stop_mode", 0)),
        "stop_pct": float(g.get("stop_pct", 2.0)),
        ## Direction inversion: 1 trades AGAINST this sleeve's own signal -- the only
        ## lever that makes it LOSE rather than flatten (2.644 -> -3.636). 0 = production.
        "signal_invert": int(g.get("signal_invert", 0)),
    }

    tfs = [int(t) for t in payload["timeframes_minutes"]]
    assert len(pairs) == int(payload["universe"]["n_pairs"]), \
        (len(pairs), payload["universe"]["n_pairs"])

    parameter_dict = {}
    for idx, tf in enumerate(tfs):
        p = payload["params_per_TF"][str(tf)]
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(p["annf"])) < 0.01, (tf, annf, p["annf"])
        pid = start_parent_id + PAIR_LEAD_LAG_PARENT_ID_STRIDE * idx
        parameter_dict[pid] = PAIR_LEAD_LAG_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            coin2=coin2,
            tf=tf,
            agg_time=f"{tf}T",
            nbdev=float(p["nbdev"]),
            z_thr=float(p["z_thr"]),
            x_atr=float(p["x_atr"]),
            c_thr=float(p["c_thr"]),
            annf=annf,
            strategy_config=strategy_config,
        )
    return parameter_dict


def plain_symbol(sym):
    """EXCHANGE-INTERNAL name -> PLAIN symbol ('BTC-USDT.PERP' -> 'BTCUSDT').

    Two naming conventions meet in this sleeve and must not be confused:
      * the client config's coin_param and `submodel_parameters` carry INTERNAL names,
        which is also what ohlcv_data is keyed by — that is the market-data language;
      * the frozen artifacts (qr31_cell_constants.parquet) are keyed by PLAIN symbols,
        because the EOD builders read `{SYM}PERP-1m-data.parquet` — that is the
        artifact language.
    Everything downstream of the parameter tuple speaks the market-data language, so
    the conversion happens at the artifact boundary. Already-plain input passes through
    unchanged, which keeps the frozen-bundle path and the offline harnesses working
    untouched.

    config_object is imported lazily: this module is deliberately importable (and
    unit-testable) without the prod config chain.
    """
    from config.config_read import config_object
    mapping = getattr(config_object, 'symbol_mapping', None)
    if not mapping:
        return sym                      # local/test mode: names are already plain
    if sym in mapping:
        return sym                      # already plain
    inverse = {v: k for k, v in mapping.items()}
    assert len(inverse) == len(mapping), 'symbol_mapping is not 1:1; cannot invert'
    assert sym in inverse, f'{sym!r} is in neither side of config_object.symbol_mapping'
    return inverse[sym]


def build_pair_lead_lag_parameter_dict_from_db(db_rows, coin1, coin2, sleeve_paths):
    """Build {parent_stratid: PAIR_LEAD_LAG_PARAM_TUPLE} for ONE pair from its
    `submodel_parameters` rows (is_live = 1) — in LIVE mode the DB is the single source
    of truth for parent ids AND kernel parameters; the frozen bundle is NOT required on
    the live host. Artifact paths ride in via the client config's sleeve_config block.

    db_rows: {parent_trading_model: model_parameters} as dumped by
    crypto-infra/db_scripts (get_dump_dict_pairs_lead_lag_som).
    """
    params_path = sleeve_paths["params_path"]
    strategy_config = None
    parameter_dict = {}
    for pid in sorted(db_rows):
        mp = db_rows[pid]
        assert mp["strategy_name"] == "pair_lead_lag", (pid, mp["strategy_name"])
        assert (mp["coin1"], mp["coin2"]) == (coin1, coin2), (pid, mp["coin1"], mp["coin2"])
        assert int(mp["long_only"]) == 1, (pid, "QR31 is a LONG-only sleeve")

        if strategy_config is None:
            strategy_config = {
                "asset": coin1,
                "pair_asset": coin2,
                "long_only": 1,
                "VT": float(mp["vt_vol_target"]),
                "txn_cost": float(mp["fixed_cost_per_trade"]),
                "slippage_per_leg_per_turn": float(mp["slippage_per_leg_per_turn"]),
                "EMA_SPAN": int(mp["ema_span"]),
                "Z_ATR_LEN": int(mp["z_atr_len"]),
                "Z_EMA_LEN": int(mp["z_ema_len"]),
                "entry_z_gate": float(mp["entry_z_gate"]),
                "z_projection_fillna": float(mp["z_projection_fillna"]),
                "corr_lookbacks": [int(x) for x in mp["corr_lookbacks"]],
                "corr_fillna": float(mp["corr_fillna"]),
                "atr_band_fast_len": int(mp["atr_band_fast_len"]),
                "atr_band_slow_len": int(mp["atr_band_slow_len"]),
                "sizing_vol_ewm_span": int(mp["sizing_vol_ewm_span"]),
                "MULT_CAP": float(mp["mult_cap"]),
                "alloc_outer_clip": [float(x) for x in mp["alloc_outer_clip"]],
                "sec_mode": int(mp["sec_mode"]),
                "sec_delay_min": int(mp["sec_delay_min"]),
                "cooldown_min": int(mp["cooldown_min"]),
                "arm_expiry_min": int(mp["arm_expiry_min"]),
                "arm_always": int(mp["arm_always"]),
                "sec_off": int(mp["sec_off"]),
                "clock_minutes": int(mp["clock_minutes"]),
                "use_override": int(mp["use_override"]),
                "rearm_off": int(mp["rearm_off"]),
                "entry_c_threshold": float(mp["entry_c_threshold"]),
                "profit_target_tp": float(mp["profit_target_tp"]),
                "calibration_window": list(mp["calibration_window"]),
                "CALIBRATION_END": str(mp["calibration_end"]),
                "annualization_days_PER": int(mp["annualization_days_per"]),
                "params_path": params_path,      # provenance string; never opened here
                "data_dir": f"{os.path.dirname(params_path)}/data",
                "activity_counts_path": sleeve_paths["activity_counts_path"],
                "qr31_cell_constants_path": sleeve_paths["qr31_cell_constants_path"],
                "mult_daily_path": sleeve_paths["mult_daily_path"],
                ## z_atr stop selector. .get() with a default, NOT hard indexing: the live
                ## submodel_parameters rows predate these keys.
                "stop_mode": int(mp.get("stop_mode", 0)),
                "stop_pct": float(mp.get("stop_pct", 2.0)),
                "signal_invert": int(mp.get("signal_invert", 0)),
            }

        tf = int(mp["tf"])
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(mp["annf"])) < 1e-6, (pid, annf, mp["annf"])

        parameter_dict[pid] = PAIR_LEAD_LAG_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            coin2=coin2,
            tf=tf,
            agg_time=mp["agg_time"],
            nbdev=float(mp["nbdev"]),
            z_thr=float(mp["z_thr"]),
            x_atr=float(mp["x_atr"]),
            c_thr=float(mp["c_thr"]),
            annf=annf,
            strategy_config=strategy_config,
        )

    assert len(parameter_dict) == len(db_rows), (len(parameter_dict), len(db_rows))
    return parameter_dict


class PairLeadLagBaseClass:
    logger_name = ""
