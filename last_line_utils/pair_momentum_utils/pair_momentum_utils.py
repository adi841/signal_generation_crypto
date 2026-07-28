
import json
import math
import os
from collections import namedtuple

## B1_v8_v10 per-CELL parameters. ONE instance == ONE (pair, TF) sleeve — the
## granularity at which the research produces an independent daily-return stream and
## the live framework runs one aggregator + one kernel -> one signal. 5 TFs per pair.
## Long-only Keltner breakout on the leg1/leg2 ratio. NOTE vs B1_v8_v9: there is NO
## daily dominance gate in v8_v10; entry sizing instead carries a cross-pair daily
## market-vol multiplier R (params.json risk_multiplier_R), read from the nightly
## artifact via strategy_config['r_state_path'].
## Per-TF tunables (K/X/Z/EZ/TT/T1/DDE/annf) ride the tuple top-level (kernel scalars);
## shared globals ride strategy_config — everything is sourced from
## pair_momentum_production/params.json, nothing hardcoded in code.
PAIR_MOMENTUM_PARAM_TUPLE = namedtuple("pair_momentum_param_tuple", [
    "parent_stratid",
    "coin1", "coin2",       # leg1 (major) / leg2 — ratio = coin1/coin2
    "tf",                   # timeframe minutes (5/15/30/60/120)
    "agg_time",             # f"{tf}T" (OHLCV aggregator format)
    "K",                    # Keltner band width; also the anti-chase numerator w=clip(K/zz,0,1)
    "X",                    # hard-stop distance in atr_eq units below the ENTRY MEDIAN
    "Z",                    # hard-stop momentum gate (stop arms only while zmed < Z)
    "EZ",                   # entry momentum floor (entry needs zmed > EZ)
    "TT",                   # develop threshold on pan, in TRAIL-VOL units (not a partial take)
    "T1",                   # undeveloped time stop, in CANDLES
    "DDE",                  # drawdown-from-peak trail multiplier, in TRAIL-VOL units
    "annf",                 # sqrt(PER*24*60/tf) — derived exact, asserted vs params.json
    "strategy_config",      # nested dict: globals + costs + artifact paths
    ## Execution route for this cell. NOT a strategy parameter and NOT in params.json --
    ## it is owned by submodel_parameters, so it is None on the frozen-bundle path (replay
    ## returns from log_dump_data before anything reads it) and an int in LIVE.
    ## live.py reads it as `getattr(tup, 'exec_type', None)`; before this field existed that
    ## getattr silently resolved to None on EVERY cell, and live_signals.execution_type is
    ## NOT NULL -- so the per-minute insert failed for the whole sleeve, caught and logged
    ## as an error by log_dump_data rather than surfacing. pair_relative_value and
    ## pair_lead_lag still have the same gap.
    "exec_type",
])

## Per-bar kernel inputs assembled by the live last-line object each minute (features
## refresh on tf-bar close). Field order mirrors the per-bar argument order of
## `cryptopairs_b1v8v10_long_iact`; the per-cell scalars T1/TT/DDE/GRACE/K/X/Z/EZ come
## off PAIR_MOMENTUM_PARAM_TUPLE at the call site.
##
## Scope: this tuple carries ONLY what the last-line object actually derives — the
## RATIO-side quantities plus costs. The four per-leg price arguments the kernel also
## takes (next_close1/2, same_close1/2) are deliberately ABSENT: live.py owns the legs
## and passes them straight to the kernel from its INPUT_DATA_TUPLE, so carrying them
## here would create a second, always-overridden source of truth.
##
## vs B1_v8_v9: the zM/zm daily-dominance fields are gone — v8_v10 has no dominance gate.
PAIR_MOMENTUM_NUMBA_INPUT_TUPLE = namedtuple('PairMomentumNumbaInputTup', [
    'p1',               # 1 when a tf-bar just closed, else 0
    'minutely_low',     # ratio minutely low (intrabar hard stop + can_new re-arm)
    'spc',              # ratio tf-candle close of the LAST COMPLETED candle
    'med',              # median20 of ratio tf-closes
    'upper',            # med + K*atr_eq (breakout trigger)
    'atr',              # atr_eq = close * ewmstd(lr, span=14)
    'lv',               # long_vol = ewmstd(lr, span=100)
    'zmed',             # z_median_rv (ewm span=20, adjust=False)
    'txn_cost',         # FIXED_COST_per_trade
    'slippage',         # per-pair slip = mean of the two legs' per-symbol slippage
    'allocation',       # clip(VT / rv_alloc / annf, 0, 1) * R_daily  (clip BEFORE R -> <= 1.75)
])


## Authoritative frozen production bundle for THIS sleeve. Named by sleeve, not by
## research version — the version it pins is declared inside the file ("strategy":
## "B1_v8_v10"). Distinct from B1_v8_v9_LONG_production/, which the pair_lead_lag
## sleeve reads; the two are independent and nothing reads across.
## Mirrors crypto_sims/PRODUCTION (config.py B1 block + engines/pairs_momentum.py);
## every value in it is asserted equal to that source.
PAIR_MOMENTUM_PARAMS_PATH = "/home/rocky/crypto_sims/pair_momentum_production/params.json"

## Sleeve name, used as the first level of the debug output paths:
##     entry_data_dir/{SLEEVE}/{coin1}_{coin2}/{parent_stratid}.csv
##     numpy_pandas_matching/{SLEEVE}/{coin1}_{coin2}/{parent_stratid}_numpy.parquet
## Both top-level dirs are shared by all five sleeves, and parent_stratids are per-pair
## LOCAL (see the band comment below), so the path needs BOTH levels to be unique: the
## sleeve name makes the layout self-describing and survives any future band overlap,
## the pair level stops two pairs of the same sleeve overwriting each other.
SLEEVE = "pair_momentum"

## Where the EOD builders write. Resolved from this file's location rather than hardcoded,
## so a checkout on another host works without editing paths.
EOD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eod_scripts", "pair_momentum", "output")

## parent_stratid band map: 1xxxxxxx directional_momentum, 2xxxxxxx retired (dead
## B1_v8_v9 rows at 20000001..20000006), 3xxxxxxx pair_relative_value (previously legacy
## hjson pair_reversal — those DB rows are dead), 4xxxxxxx pair_momentum (previously
## legacy pair_breakout — dead), 5xxxxxxx pair_lead_lag, 6/7xxxxxxx legacy single-asset,
## 8/9xxxxxxx free. IDs are per-pair LOCAL (every pair reuses 40000001/03/05/07/09) —
## global uniqueness is applied downstream when registering child_id_info.
## STRIDE 2: the house convention for single-entry PAIR strategies reserves 2 ids per
## (pair, TF) cell — the signal stream posts at +0 and the +1 slot is the leg-2 id
## used downstream at child-id registration. (2-entry pair strategies reserve 4.)
PAIR_MOMENTUM_START_PARENT_ID = 40000001
PAIR_MOMENTUM_PARENT_ID_STRIDE = 2


def build_pair_momentum_parameter_dict(coin1, coin2, params_path=PAIR_MOMENTUM_PARAMS_PATH,
                                       start_parent_id=PAIR_MOMENTUM_START_PARENT_ID):
    """Build {parent_stratid: PAIR_MOMENTUM_PARAM_TUPLE} for ONE pair — all 5 TF cells.

    One tuple == one (pair, TF) sleeve; parent_stratids are per-pair local, assigned
    `start_parent_id + 2*tf_index` (stride 2 — single-entry pair convention: the +1
    slot is the leg-2 id, there is no second entry). Everything is read from
    params.json: per-TF K/X/Z/EZ/TT/T1/DDE,
    global constants, per-symbol slippage (pair slip = mean of legs). annf is derived
    exact from annualization_days_PER (params.json stores 2dp-rounded values) and
    asserted against the json. Pure / side-effect free so the matching harness can use
    it without the live Base/socket machinery.

    The two EOD-built artifacts are surfaced as explicit paths (`rv_alloc_bounds_path`,
    `r_state_path`) rather than resolved off `data_dir`; R's own constants are frozen in
    params.json under global_constants / risk_multiplier_R and are consumed by the
    builder, not here.
    """
    with open(params_path) as f:
        payload = json.load(f)

    g = payload["global_constants"]
    slip_map = payload["slippage_per_symbol_fraction"]
    pairs = [tuple(p) for p in payload["universe"]["pairs_leg1_leg2"]]
    if (coin1, coin2) not in pairs:
        raise KeyError(f"({coin1}, {coin2}) not in the B1_v8_v10 universe ({len(pairs)} pairs). "
                       f"Note B1 excludes BNB/LTC/BCH/TRX/NEAR/ATOM — it is not the QR1/QR31 universe.")

    slippage = (float(slip_map[coin1]) + float(slip_map[coin2])) / 2.0
    data_dir = f"{os.path.dirname(params_path)}/data"

    ## The two EOD-built artifacts live together in eod_scripts/pair_momentum/output/ and
    ## are surfaced as EXPLICIT paths rather than being resolved off `data_dir` (the
    ## convention pair_relative_value uses for entry_score_model_path). Two reasons:
    ##   * rv_alloc_bounds there has all 85 (pair, TF) cells; the copy under data_dir has
    ##     only the 5 BTC/AVAX rows built during test 1, and load_rv_alloc_bounds hard-raises
    ##     on a missing cell — so 16 of the 17 pairs could not start at all.
    ##   * r_state_daily is REWRITTEN NIGHTLY, so it is not a frozen bundle artifact at all;
    ##     see last_line_utils/pair_momentum_utils/r_state.py for the polling reader.

    strategy_config = {
        "asset": coin1,
        "pair_asset": coin2,
        "long_only": 1,
        "VT": float(g["VT_vol_target"]),
        "GRACE": float(g["GRACE"]),
        "txn_cost": float(g["FIXED_COST_per_trade"]),
        "slippage_per_leg_per_turn": slippage,
        "median_window_bars": int(g["median_window_bars"]),
        "atr_ewm_span": int(g["atr_ewm_span"]),
        "long_vol_ewm_span": int(g["long_vol_ewm_span"]),
        "rv_ewm_span": int(g["rv_ewm_span"]),
        "rv_scale": float(g["rv_scale"]),
        "zmed_ewm_span": int(g["zmed_ewm_span"]),
        "zmed_ewm_adjust": bool(g["zmed_ewm_adjust"]),
        "rv_alloc_clip_quantiles": list(g["rv_alloc_clip_quantiles"]),
        "rv_alloc_calibration_window": list(g["rv_alloc_calibration_window"]),
        "annualization_days_PER": int(g["annualization_days_PER"]),
        "params_path": params_path,
        "data_dir": data_dir,
        "rv_alloc_bounds_path": f"{EOD_OUTPUT_DIR}/rv_alloc_bounds.parquet",
        "r_state_path": f"{EOD_OUTPUT_DIR}/r_state_daily.parquet",
    }

    tfs = [int(t) for t in payload["timeframes_minutes"]]
    assert len(pairs) == int(payload["universe"]["n_pairs"]), \
        (len(pairs), payload["universe"]["n_pairs"])

    parameter_dict = {}
    for idx, tf in enumerate(tfs):
        p = payload["params_per_TF"][str(tf)]
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(p["annf"])) < 0.01, (tf, annf, p["annf"])
        pid = start_parent_id + PAIR_MOMENTUM_PARENT_ID_STRIDE * idx
        parameter_dict[pid] = PAIR_MOMENTUM_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            coin2=coin2,
            tf=tf,
            agg_time=f"{tf}T",
            K=float(p["K"]),
            X=float(p["X"]),
            Z=float(p["Z"]),
            EZ=float(p["EZ"]),
            TT=float(p["TT"]),
            T1=float(p["T1"]),
            DDE=float(p["DDE"]),
            annf=annf,
            strategy_config=strategy_config,
            exec_type=None,     # bundle has no execution route; see the field comment
        )
    return parameter_dict


def plain_symbol(sym):
    """EXCHANGE-INTERNAL name -> PLAIN symbol ('BTC-USDT.PERP' -> 'BTCUSDT').

    Two naming conventions meet in this sleeve and must not be confused:
      * the client config's coin_param and `submodel_parameters` carry INTERNAL names,
        which is also what ohlcv_data is keyed by — that is the market-data language;
      * the frozen artifacts (rv_alloc_bounds.parquet) and params.json are keyed by PLAIN
        symbols, because the EOD builders read `{SYM}PERP-1m-data.parquet` — that is the
        artifact language.
    The conversion happens once, at the artifact boundary. Already-plain input passes
    through unchanged, which keeps the frozen-bundle path and the offline harnesses
    working untouched.

    Twin of pair_relative_value_utils.plain_symbol. Deliberately duplicated rather than
    imported: the sleeves are independent and nothing reads across, and a shared copy
    would put a verified sleeve at risk for a 10-line pure helper.

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


def build_pair_momentum_parameter_dict_from_db(db_rows, coin1, coin2, sleeve_paths):
    """Build {parent_stratid: PAIR_MOMENTUM_PARAM_TUPLE} for ONE pair from its
    `submodel_parameters` rows (is_live = 1) — in LIVE mode the DB is the single source
    of truth for parent ids AND kernel parameters; the frozen bundle is NOT required on
    the live host. Artifact paths ride in via the client config's sleeve_config block.

    db_rows: {parent_trading_model: model_parameters} as dumped by
    crypto-infra/db_scripts (get_dump_dict_pairs_momentum_som).
    """
    params_path = sleeve_paths["params_path"]
    strategy_config = None
    parameter_dict = {}
    for pid in sorted(db_rows):
        mp = db_rows[pid]
        assert mp["strategy_name"] == "pair_momentum", (pid, mp["strategy_name"])
        assert (mp["coin1"], mp["coin2"]) == (coin1, coin2), (pid, mp["coin1"], mp["coin2"])
        assert int(mp["long_only"]) == 1, (pid, "B1 is a LONG-only sleeve")

        if strategy_config is None:
            strategy_config = {
                "asset": coin1,
                "pair_asset": coin2,
                "long_only": 1,
                "VT": float(mp["vt_vol_target"]),
                "GRACE": float(mp["grace"]),
                "txn_cost": float(mp["fixed_cost_per_trade"]),
                "slippage_per_leg_per_turn": float(mp["slippage_per_leg_per_turn"]),
                "median_window_bars": int(mp["median_window_bars"]),
                "atr_ewm_span": int(mp["atr_ewm_span"]),
                "long_vol_ewm_span": int(mp["long_vol_ewm_span"]),
                "rv_ewm_span": int(mp["rv_ewm_span"]),
                "rv_scale": float(mp["rv_scale"]),
                "zmed_ewm_span": int(mp["zmed_ewm_span"]),
                "zmed_ewm_adjust": bool(mp["zmed_ewm_adjust"]),
                "rv_alloc_clip_quantiles": list(mp["rv_alloc_clip_quantiles"]),
                "rv_alloc_calibration_window": list(mp["rv_alloc_calibration_window"]),
                "annualization_days_PER": int(mp["annualization_days_per"]),
                "params_path": params_path,      # provenance string; never opened here
                "data_dir": f"{os.path.dirname(params_path)}/data",
                "rv_alloc_bounds_path": sleeve_paths["rv_alloc_bounds_path"],
                "r_state_path": sleeve_paths["r_state_path"],
            }

        tf = int(mp["tf"])
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(mp["annf"])) < 1e-6, (pid, annf, mp["annf"])

        parameter_dict[pid] = PAIR_MOMENTUM_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            coin2=coin2,
            tf=tf,
            agg_time=mp["agg_time"],
            K=float(mp["k"]),
            X=float(mp["x"]),
            Z=float(mp["z"]),
            EZ=float(mp["ez"]),
            TT=float(mp["tt"]),
            T1=float(mp["t1"]),
            DDE=float(mp["dde"]),
            annf=annf,
            strategy_config=strategy_config,
            ## NOT NULL downstream in live_signals -- fail here, where the offending pid is
            ## named, rather than at the insert.
            exec_type=int(mp["exec_type"]),
        )

    assert len(parameter_dict) == len(db_rows), (len(parameter_dict), len(db_rows))
    return parameter_dict


class PairMomentumBaseClass:
    logger_name = ""
