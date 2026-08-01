
import json
import math
import os
from collections import namedtuple

## QR1_v4 per-CELL parameters. ONE instance == ONE (pair, TF) sleeve — the granularity
## at which the research produces an independent daily-return stream and the live
## framework runs one aggregator + one kernel -> one signal. 5 TFs per pair.
## SHORT-only relative-value fade on the leg1/leg2 ratio: short the ratio on a gated
## band breach, cover on reversion. TWO tranches per trade (entry + one scale-in at
## SEC_MULT x ATR14 spacing), each half the cell allocation — so unlike B1 the
## reversal2/QR31 dual-stream contract (res2 / order_tag2) DOES apply to this sleeve.
## Per-TF tunables (z_entry/z_thr/nbdev/x_atr/annf) ride the tuple top-level (kernel
## scalars); shared globals ride strategy_config — everything is sourced from
## pair_relative_value_production/params.json, nothing hardcoded in code.
PAIR_RELATIVE_VALUE_PARAM_TUPLE = namedtuple("pair_relative_value_param_tuple", [
    "parent_stratid",
    "coin1", "coin2",       # leg1 (major) / leg2 — ratio = coin1/coin2
    "tf",                   # timeframe minutes (15/30/60/120/240)
    "agg_time",             # f"{tf}T" (OHLCV aggregator format)
    "z_entry",              # entry momentum gate: entry requires zn > z_entry
    "z_thr",                # stop momentum gate: the z/ATR stop needs zn > z_thr
    "nbdev",                # band width: upper = min(centre + nbdev*ATR14, centre + nbdev*ATR50)
    "x_atr",                # stop distance above the FIRST entry ratio, in clipped-ATR50 % units
    "annf",                 # sqrt(PER*24*60/tf) — derived exact, asserted vs params.json
    "strategy_config",      # nested dict: globals + costs + artifact paths
    "exec_type",            # live_signals.execution_type — from the DB row; see below
])
## `exec_type` is the broker's execution style for the cell and lands in
## live_signals.execution_type, which is NOT NULL. It exists only in the DB
## (submodel_parameters.model_parameters), never in the frozen bundle, so it defaults to
## None for the hist_replay builder — replay writes no DB rows, so None never reaches
## postgres from that path. live.py reads it off the tuple (`getattr(tup, 'exec_type')`).
PAIR_RELATIVE_VALUE_PARAM_TUPLE.__new__.__defaults__ = (None,)

## Per-bar kernel inputs assembled by the live last-line object each minute (candle
## features refresh on tf-bar close). Field order mirrors the per-bar argument order of
## the QR1 kernel (`qr1_instr`); the per-cell scalars z_entry/z_thr/x_atr and the
## globals SEC_MULT/C_THR come off PAIR_RELATIVE_VALUE_PARAM_TUPLE at the call site,
## as do the two kernel constants m2=1 (msig regime convention) and tp=1000.0 (the
## vestigial profit-target input — constant in the frozen reference).
##
## Scope: this tuple carries ONLY what the last-line object actually derives — the
## RATIO-side quantities plus costs. The four per-leg price arguments the kernel also
## takes (next_close1/2, same_close1/2) are deliberately ABSENT: live.py owns the legs
## and passes them straight to the kernel from its INPUT_DATA_TUPLE, so carrying them
## here would create a second, always-overridden source of truth.
PAIR_RELATIVE_VALUE_NUMBA_INPUT_TUPLE = namedtuple('PairRelativeValueNumbaInputTup', [
    'p1',               # 1 when a tf-bar just closed, else 0
    'minutely_high',    # ratio minutely high (breach test, scale-in spacing, z/ATR stop, dist_bp)
    'minutely_low',     # ratio minutely low (middle-touch re-arm, profit exit, tsm counter)
    'msig',             # +1 iff (zn > z_entry) AND skew gate AND wide-band gate, else -1
    'upper',            # active upper band, price units (last completed candle)
    'middle',           # active channel centre (EMA10 adjust=True of candle closes)
    'lower',            # lower band — NEVER read by the short kernel; kernel-arg parity only
    'atr',              # clipped ATR14 in % of close (scale-in spacing unit)
    'atr2',             # clipped ATR50 in % of close (z/ATR stop distance unit)
    'zn',               # v4 engine z (entry AND exit; adjust=False EMAs over price-unit EWM vol)
    'corr',             # mean rolling leg corr — gate DELETED in v4 (C_THR=-1), parity only
    'skew_ok',          # skew gate bool (msig component + kernel instrumentation arg)
    'txn_cost',         # FIXED_COST_per_trade (per fill per tranche)
    'slippage',         # per-pair slip = mean of the two legs' per-symbol slippage
    'allocation',       # al_ew(EV10 vol-target) x clip(score/med, SC_CLIP)  [TODO(EXPO): x expo_daily]
])


## Authoritative frozen production bundle for THIS sleeve. Named by sleeve, not by
## research version — the version it pins is declared inside the file ("strategy":
## "QR1_v4"). Mirrors crypto_sims/PRODUCTION (config.py QR1 block + engines/
## pairs_relvalue.py + engines/core/vendored_qr1.py + kernel_qr1.py); every value in
## it was asserted equal to that source at bundle creation (77/77 check).
PAIR_RELATIVE_VALUE_PARAMS_PATH = "/home/rocky/crypto_sims/pair_relative_value_production/params.json"

## parent_stratid band map: 1xxxxxxx directional_momentum, 2xxxxxxx retired (dead
## B1_v8_v9 rows at 20000001..20000006), 3xxxxxxx pair_relative_value (previously legacy
## hjson pair_reversal — those DB rows are dead), 4xxxxxxx pair_momentum (previously
## legacy pair_breakout — dead), 5xxxxxxx pair_lead_lag, 6/7xxxxxxx legacy single-asset,
## 8/9xxxxxxx free. IDs are per-pair LOCAL — global uniqueness is applied downstream
## when registering child_id_info.
## STRIDE 4 per TF (the reversal2 dual-stream odd-id convention): the PRIMARY
## (tranche-1) id sits at +0 and live.py posts the SCALE-IN (tranche-2) DB stream at
## parent_id + 2, so the five TF cells claim 30000001/05/09/13/17 with tranche-2
## streams at 30000003/07/11/15/19 (+1/+3 spare) — collision-free by construction.
PAIR_RELATIVE_VALUE_START_PARENT_ID = 30000001
PAIR_RELATIVE_VALUE_PARENT_ID_STRIDE = 4


def build_pair_relative_value_parameter_dict(coin1, coin2,
                                             params_path=PAIR_RELATIVE_VALUE_PARAMS_PATH,
                                             start_parent_id=PAIR_RELATIVE_VALUE_START_PARENT_ID):
    """Build {parent_stratid: PAIR_RELATIVE_VALUE_PARAM_TUPLE} for ONE pair — all 5 TF cells.

    One tuple == one (pair, TF) sleeve; parent_stratids are per-pair local, assigned
    `start_parent_id + 4 * tf_index` (stride 4: live.py posts the scale-in tranche's
    DB stream at parent_id + 2, the reversal2 convention). Everything is read from
    params.json: per-TF z_entry/z_thr/nbdev/x_atr, global constants, per-symbol
    slippage (pair slip = mean of legs). annf is derived exact from
    annualization_days_PER (params.json stores 2dp-rounded values) and asserted
    against the json. Pure / side-effect free so the matching harness can use it
    without the live Base/socket machinery.

    NOTE: the EXPO overlay multiplier is deliberately NOT surfaced in strategy_config
    yet — it is a cross-market daily state that has to arrive via a shared artifact
    (data/expo_daily.parquet), and it is wired in the sizing step. Its frozen
    constants are already in params.json (global_constants + expo_overlay). Same
    policy as pair_momentum's R: no silent default to 1.0. The per-cell IS constants
    (kv sizing scale, wmed wide-band median, ATR clip bounds) live in
    data/qr1_cell_constants.parquet and are loaded by their own loader in the
    generate_signal step — only the paths ride strategy_config here.
    """
    with open(params_path) as f:
        payload = json.load(f)

    g = payload["global_constants"]
    slip_map = payload["slippage_per_symbol_fraction"]
    pairs = [tuple(p) for p in payload["universe"]["pairs_leg1_leg2"]]
    if (coin1, coin2) not in pairs:
        raise KeyError(f"({coin1}, {coin2}) not in the QR1_v4 universe ({len(pairs)} pairs). "
                       f"Note QR1 includes BNB but excludes DOT/UNI — it is not the "
                       f"B1/QR31 universe, and leg1 must be BTCUSDT or ETHUSDT.")

    slippage = (float(slip_map[coin1]) + float(slip_map[coin2])) / 2.0
    data_dir = f"{os.path.dirname(params_path)}/data"

    strategy_config = {
        "asset": coin1,
        "pair_asset": coin2,
        "short_only": 1,
        "VT": float(g["VT_vol_target"]),
        "txn_cost": float(g["FIXED_COST_per_trade"]),
        "slippage_per_leg_per_turn": slippage,
        # ---- sizing (EV10 fast vol; per-cell kv scale + wmed gate median are frozen
        # in data/qr1_cell_constants.parquet, loaded in the generate_signal step) ----
        "sizing_vol_ewm_span": int(g["sizing_vol_ewm_span"]),
        # ---- kernel globals ----
        "SEC_MULT": float(g["SEC_MULT"]),
        "C_THR": float(g["C_THR"]),
        # ---- entry-quality score ----
        "SC_CLIP": [float(x) for x in g["SC_CLIP"]],
        "entry_score_model_path": f"{data_dir}/relvalue_entry_score_model.json",
        # ---- EXPO overlay constants (artifact wiring in the sizing step) ----
        "EXPO_FLOOR": float(g["EXPO_FLOOR"]),
        "EXPO_SLOPE": float(g["EXPO_SLOPE"]),
        "EXPO_WIN": int(g["EXPO_WIN"]),
        # ---- band / z / gate construction ----
        "band_centre_ema_span": int(g["band_centre_ema_span"]),
        "band_constr": str(g["band_constr"]),
        "atr_band_fast_len": int(g["atr_band_fast_len"]),
        "atr_band_slow_len": int(g["atr_band_slow_len"]),
        "atr_alloc_legacy_len": int(g["atr_alloc_legacy_len"]),
        "zn_centre_ema_span": int(g["zn_centre_ema_span"]),
        "zn_vol_ewm_span": int(g["zn_vol_ewm_span"]),
        "zn_smooth_ewm_span": int(g["zn_smooth_ewm_span"]),
        "zn_denom_floor": float(g["zn_denom_floor"]),
        "zn_projection_fillna": float(g["zn_projection_fillna"]),
        "SKEW_N": int(g["SKEW_N"]),
        "corr_lookbacks": [int(x) for x in g["corr_lookbacks"]],
        # ---- frozen calibration window (IS-constants provenance; never recompute) ----
        "rv_calibration_window": list(g["rv_calibration_window"]),
        "CALIBRATION_END": str(g["CALIBRATION_END"]),
        "annualization_days_PER": int(g["annualization_days_PER"]),
        "params_path": params_path,
        "data_dir": data_dir,
    }

    tfs = [int(t) for t in payload["timeframes_minutes"]]
    assert len(pairs) == int(payload["universe"]["n_pairs"]), \
        (len(pairs), payload["universe"]["n_pairs"])
    assert sorted(str(t) for t in tfs) == sorted(payload["params_per_TF"].keys()), \
        (tfs, list(payload["params_per_TF"].keys()))

    parameter_dict = {}
    for idx, tf in enumerate(tfs):
        p = payload["params_per_TF"][str(tf)]
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(p["annf"])) < 0.01, (tf, annf, p["annf"])
        pid = start_parent_id + PAIR_RELATIVE_VALUE_PARENT_ID_STRIDE * idx
        parameter_dict[pid] = PAIR_RELATIVE_VALUE_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            coin2=coin2,
            tf=tf,
            agg_time=f"{tf}T",
            z_entry=float(p["z_entry"]),
            z_thr=float(p["z_thr"]),
            nbdev=float(p["nbdev"]),
            x_atr=float(p["x_atr"]),
            annf=annf,
            strategy_config=strategy_config,
        )
    return parameter_dict


def plain_symbol(sym):
    """EXCHANGE-INTERNAL name -> PLAIN symbol ('BTC-USDT.PERP' -> 'BTCUSDT').

    Two naming conventions meet in this sleeve and must not be confused:
      * the client config's coin_param and `submodel_parameters` carry INTERNAL names,
        which is also what ohlcv_data is keyed by — that is the market-data language;
      * the frozen artifacts (qr1_cell_constants.parquet) are keyed by PLAIN symbols,
        because the EOD builders read `{SYM}PERP-1m-data.parquet` — that is the
        artifact language.
    Everything downstream of the parameter tuple speaks the artifact language, so the
    conversion happens once, here. Already-plain input passes through unchanged, which
    keeps the frozen-bundle path and the offline harnesses working untouched.

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


def build_pair_relative_value_parameter_dict_from_db(db_rows, coin1, coin2, sleeve_paths):
    """Build {parent_stratid: PAIR_RELATIVE_VALUE_PARAM_TUPLE} for ONE pair from its
    `submodel_parameters` rows (is_live = 1) — in LIVE mode the DB is the single source
    of truth for parent ids AND kernel parameters; the frozen bundle is NOT required on
    the live host. Artifact paths ride in via the client config's sleeve_config block:
    data_dir is derived from qr1_cell_constants_path (the score model, cell constants
    and expo_daily all live in the same bundle data/ directory).

    db_rows: {parent_trading_model: model_parameters} as dumped by
    crypto-infra/db_scripts (get_dump_dict_pairs_relative_value_som). The tranche-2 DB
    stream stays at parent_id + 2 of each base row.
    """
    params_path = sleeve_paths["params_path"]
    data_dir = os.path.dirname(sleeve_paths["qr1_cell_constants_path"])
    assert os.path.dirname(sleeve_paths["expo_daily_path"]) == data_dir, \
        (sleeve_paths["qr1_cell_constants_path"], sleeve_paths["expo_daily_path"])
    strategy_config = None
    parameter_dict = {}
    for pid in sorted(db_rows):
        mp = db_rows[pid]
        assert mp["strategy_name"] == "pair_relative_value", (pid, mp["strategy_name"])
        assert (mp["coin1"], mp["coin2"]) == (coin1, coin2), (pid, mp["coin1"], mp["coin2"])
        assert int(mp["long_only"]) == 0, (pid, "QR1 is a SHORT-only sleeve")

        if strategy_config is None:
            strategy_config = {
                "asset": coin1,
                "pair_asset": coin2,
                "short_only": 1,
                "VT": float(mp["vt_vol_target"]),
                "txn_cost": float(mp["fixed_cost_per_trade"]),
                "slippage_per_leg_per_turn": float(mp["slippage_per_leg_per_turn"]),
                "sizing_vol_ewm_span": int(mp["sizing_vol_ewm_span"]),
                "SEC_MULT": float(mp["sec_mult"]),
                "C_THR": float(mp["c_thr"]),
                "SC_CLIP": [float(x) for x in mp["sc_clip"]],
                "entry_score_model_path": f"{data_dir}/relvalue_entry_score_model.json",
                "EXPO_FLOOR": float(mp["expo_floor"]),
                "EXPO_SLOPE": float(mp["expo_slope"]),
                "EXPO_WIN": int(mp["expo_win"]),
                "band_centre_ema_span": int(mp["band_centre_ema_span"]),
                "band_constr": str(mp["band_constr"]),
                "atr_band_fast_len": int(mp["atr_band_fast_len"]),
                "atr_band_slow_len": int(mp["atr_band_slow_len"]),
                "atr_alloc_legacy_len": int(mp["atr_alloc_legacy_len"]),
                "zn_centre_ema_span": int(mp["zn_centre_ema_span"]),
                "zn_vol_ewm_span": int(mp["zn_vol_ewm_span"]),
                "zn_smooth_ewm_span": int(mp["zn_smooth_ewm_span"]),
                "zn_denom_floor": float(mp["zn_denom_floor"]),
                "zn_projection_fillna": float(mp["zn_projection_fillna"]),
                "SKEW_N": int(mp["skew_n"]),
                "corr_lookbacks": [int(x) for x in mp["corr_lookbacks"]],
                "rv_calibration_window": list(mp["rv_calibration_window"]),
                "CALIBRATION_END": str(mp["calibration_end"]),
                "annualization_days_PER": int(mp["annualization_days_per"]),
                "params_path": params_path,      # provenance string; never opened here
                "data_dir": data_dir,
            }

        tf = int(mp["tf"])
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(mp["annf"])) < 1e-6, (pid, annf, mp["annf"])

        ## live_signals.execution_type is NOT NULL and the broker domain is {1,2,3,4}.
        ## Assert rather than default: a missing/rogue exec_type here would otherwise
        ## surface as a per-minute insert failure inside log_dump_data's swallow-and-log
        ## handler, i.e. silently no signals at all.
        assert mp.get("exec_type") is not None, f"{pid}: model_parameters has no exec_type"
        exec_type = int(mp["exec_type"])
        assert exec_type in (1, 2, 3, 4), f"{pid}: exec_type {exec_type} outside {{1,2,3,4}}"

        parameter_dict[pid] = PAIR_RELATIVE_VALUE_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            coin2=coin2,
            tf=tf,
            agg_time=mp["agg_time"],
            z_entry=float(mp["z_entry"]),
            z_thr=float(mp["z_thr"]),
            nbdev=float(mp["nbdev"]),
            x_atr=float(mp["x_atr"]),
            annf=annf,
            strategy_config=strategy_config,
            exec_type=exec_type,
        )

    assert len(parameter_dict) == len(db_rows), (len(parameter_dict), len(db_rows))
    return parameter_dict


class PairRelativeValueBaseClass:
    logger_name = ""
