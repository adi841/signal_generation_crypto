
import json
import math
import os
from collections import namedtuple

## DMP_v3_2 per-CELL-SIDE parameters. ONE instance == ONE (asset, TF, side) stream — the
## granularity at which the research produces an independent daily-return stream and the
## live framework runs one aggregator + one kernel -> one signal.
##
## DMP is SINGLE-ASSET: there is no ratio anywhere in this strategy, so there is no coin2.
## Each (asset, TF) cell carries TWO INDEPENDENT state machines — a long Keltner breakout
## and its mirrored short — and we model them as two separate entries here rather than one
## dual-stream object: separate parent_stratid, strategy object, kernel, order tag, and the
## standard one-exec + one-DB signal pair at each id. The two sides share every candle
## feature, so the only cost is computing those features twice per (asset, TF); that is 8
## strategy objects per asset process (pair_momentum runs 5, sa_mft runs 68).
##
## Per-TF/per-side tunables ride the tuple top-level (they are kernel scalars); shared
## globals ride strategy_config. Everything is sourced from
## directional_momentum_production/params.json — nothing is hardcoded in code.
##
## NOTE the parameter set is FLATTER than B1's: K/X/Z/EZ/TT/T1 are SHARED across all four
## TFs (2.5 / 2.5 / -0.3 / 0.0 / 6.0 / 150.0). The only per-TF quantity is the trail
## multiplier, and the only per-SIDE quantity in the whole strategy is the M factor inside
## it. There is no GRACE (B1's kern_v7 has one; kern_long/kern_short do not).
DIRECTIONAL_MOMENTUM_PARAM_TUPLE = namedtuple("directional_momentum_param_tuple", [
    "parent_stratid",
    "coin1",                # the traded asset (single-asset sleeve — there is no coin2)
    "side",                 # "LONG" or "SHORT" — selects the kernel and gates the R multiplier
    "tf",                   # timeframe minutes (5/15/30/60)
    "agg_time",             # f"{tf}T" (OHLCV aggregator format)
    "K",                    # Keltner band half-width; also the anti-chase numerator w=clip(K/zz,0,1)
    "X",                    # hard-stop distance in atr_eq units from the ENTRY MEDIAN
    "Z",                    # hard-stop momentum gate (LONG arms while zmed < Z; SHORT while zmed > -Z)
    "EZ",                   # entry momentum floor (LONG needs zmed > EZ; SHORT needs zmed < -EZ)
    "TT",                   # develop threshold on adv, in TRAIL-VOL units (NOT a partial take)
    "T1",                   # undeveloped time stop, in CANDLES
    "dde_mult",             # M_side * DDE_LAW[tf] — the ONLY parameter that differs by side
    "annf",                 # sqrt(PER*24*60/tf) — derived exact, asserted vs params.json
    "strategy_config",      # nested dict: globals + costs + artifact paths
    ## Execution route for this cell. NOT a strategy parameter and NOT in params.json --
    ## it is owned by submodel_parameters, so it is None on the frozen-bundle path (replay
    ## returns from log_dump_data before anything reads it) and an int in LIVE.
    ## live.py reads it as `getattr(tup, 'exec_type', None)`; without this field that getattr
    ## silently resolves to None on EVERY cell, and live_signals.execution_type is NOT NULL
    ## -- so the per-minute insert fails for the whole sleeve, caught and logged as an error
    ## by log_dump_data rather than surfacing. Mirrors pair_momentum, which hit this first.
    "exec_type",
])

## Per-bar kernel inputs assembled by the live last-line object each minute (features
## refresh on tf-bar close). The per-cell scalars T1/TT/dde_mult/K/X/Z/EZ come off
## DIRECTIONAL_MOMENTUM_PARAM_TUPLE at the call site.
##
## Scope: this tuple carries ONLY what the last-line object actually derives — the candle
## features plus the minute extremes and costs. The two per-minute PRICE arguments the
## kernels also take (next_close, same_close) are deliberately ABSENT: live.py owns them
## and passes them straight from its INPUT_DATA_TUPLE, so carrying them here would create a
## second, always-overridden source of truth. Same split as pair_momentum.
##
## The tuple is UNIFORM ACROSS SIDES — it carries both minute extremes and both band lines
## even though a given kernel reads only one of each. That keeps one debug/dump schema for
## the whole sleeve and lets the batch-vs-live handoff test compare all 13 fields on either
## side; the unused half costs two floats a minute.
DIRECTIONAL_MOMENTUM_NUMBA_INPUT_TUPLE = namedtuple('DirectionalMomentumNumbaInputTup', [
    'p1',               # 1 when a tf-bar just closed, else 0
    'minutely_low',     # asset minute low  (LONG intrabar hard stop + can_new re-arm)
    'minutely_high',    # asset minute high (SHORT intrabar hard stop + can_new re-arm)
    'asset_close',      # tf-candle close of the LAST COMPLETED candle (the kernels' `ac`)
    'median_line',      # median20 of tf-closes — the channel centre
    'upper_line',       # median_line + K*atr_eq (LONG breakout trigger)
    'lower_line',       # median_line - K*atr_eq (SHORT breakout trigger)
    'atr_eq',           # close * ewmstd(lr, span=10)   <-- span 10, NOT the module's 14
    'long_vol',         # ewmstd(lr, span=100) — trail vol
    'z_median',         # ewm(span=20, adjust=False).mean() of (close-median20)/atr_eq
    'txn_cost',         # FIXED_COST_per_trade, charged per fill
    'slippage',         # this asset's per-symbol slippage (single asset — no leg averaging)
    'allocation',       # clip(VT / rv_alloc / annf, 0, 1), x R_daily on the SHORT side only
])


## Authoritative frozen production bundle for THIS sleeve. Named by sleeve, not by research
## version — the version it pins is declared inside the file ("strategy": "DMP_v3_2").
## Mirrors crypto_sims/PRODUCTION (config.py DMP block + engines/directional_momentum.py +
## engines/core/vendored_b1_dmp.py); every value in it is asserted equal to that source by
## the builder that emits it.
DIRECTIONAL_MOMENTUM_PARAMS_PATH = "/home/rocky/crypto_sims/directional_momentum_production/params.json"

## Sleeve name, used as the first level of the debug output paths:
##     entry_data_dir/{SLEEVE}/{coin1}/{parent_stratid}.csv
##     numpy_pandas_matching/{SLEEVE}/{coin1}/{parent_stratid}_numpy.parquet
## Both top-level dirs are shared by every sleeve, and parent_stratids are per-ASSET LOCAL
## (see the band comment below), so the path needs BOTH levels to be unique: the sleeve name
## makes the layout self-describing and survives any future band overlap, the asset level
## stops two assets of the same sleeve overwriting each other. (The asset level replaces
## pair_momentum's {coin1}_{coin2} level — same purpose, one leg.)
SLEEVE = "directional_momentum"

## Where the EOD builders write. Resolved from this file's location rather than hardcoded,
## so a checkout on another host works without editing paths.
EOD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eod_scripts", "directional_momentum", "output")

## parent_stratid band map: 1xxxxxxx directional_momentum (was sa_directional's, which
## this sleeve replaces), 2xxxxxxx retired (dead B1_v8_v9 rows at 20000001..20000006),
## 3xxxxxxx pair_relative_value (previously legacy hjson pair_reversal — those DB rows
## are dead), 4xxxxxxx pair_momentum (previously legacy pair_breakout — dead), 5xxxxxxx
## pair_lead_lag, 6/7xxxxxxx legacy single-asset, 8/9xxxxxxx free.
##
## STRIDE 2 because each (asset, TF) cell emits TWO streams: LONG at start + 2*tf_index,
## SHORT at that +1. IDs are per-ASSET LOCAL (every asset reuses 10000001..10000008) —
## global uniqueness is applied downstream when registering child_id_info.
##
##     tf   5 : LONG 10000001   SHORT 10000002
##     tf  15 : LONG 10000003   SHORT 10000004
##     tf  30 : LONG 10000005   SHORT 10000006
##     tf  60 : LONG 10000007   SHORT 10000008
DIRECTIONAL_MOMENTUM_START_PARENT_ID = 10000001
DIRECTIONAL_MOMENTUM_PARENT_ID_STRIDE = 2

## Side order within a cell — index into the stride. Defined once here so the builder, the
## signal generator and any harness agree on which id is which without re-deriving it.
DIRECTIONAL_MOMENTUM_SIDES = ("LONG", "SHORT")


def build_directional_momentum_parameter_dict(
        coin1,
        params_path=DIRECTIONAL_MOMENTUM_PARAMS_PATH,
        start_parent_id=DIRECTIONAL_MOMENTUM_START_PARENT_ID,
        stride=DIRECTIONAL_MOMENTUM_PARENT_ID_STRIDE):
    """Build {parent_stratid: DIRECTIONAL_MOMENTUM_PARAM_TUPLE} for ONE asset.

    Returns 8 tuples: 4 TFs x 2 sides. parent_stratids are per-asset local, assigned
    `start_parent_id + stride*tf_index + side_index` (LONG first, then SHORT).

    Everything is read from params.json: the shared K/X/Z/EZ/TT/T1, the per-TF/per-side
    trail multiplier, global constants and per-symbol slippage. annf is derived exact from
    annualization_days_PER (params.json stores 2dp-rounded values) and asserted against the
    json, as is dde_mult against M_side * DDE_base. Pure / side-effect free so the matching
    harness can use it without the live Base/socket machinery.

    The three EOD-built artifacts are surfaced as explicit paths rather than resolved off
    `data_dir`, for the same two reasons pair_momentum does it: the bundle's data/ copy only
    ever holds the cells built during a test run (and the loaders hard-raise on a missing
    cell, so every other asset would fail to start), and r_state_daily is REWRITTEN NIGHTLY
    so it is not a frozen bundle artifact at all.
    """
    with open(params_path) as f:
        payload = json.load(f)

    g = payload["global_constants"]
    slip_map = payload["slippage_per_symbol_fraction"]
    assets = list(payload["universe"]["assets"])
    if coin1 not in assets:
        raise KeyError(
            f"{coin1} not in the DMP_v3_2 universe ({len(assets)} assets: {', '.join(assets)}). "
            f"Note this is NOT the B1/QR1/QR31 universe — BNB is a DMP asset but not a B1 pair leg, "
            f"and AAVE was removed at v3.1.")
    assert len(assets) == int(payload["universe"]["n_assets"]), \
        (len(assets), payload["universe"]["n_assets"])

    ## Single asset — no leg averaging. B1/QR1 charge mean(leg1_slip, leg2_slip); DMP charges
    ## this symbol's slippage on every fill of both sides.
    slippage = float(slip_map[coin1])
    data_dir = f"{os.path.dirname(params_path)}/data"

    strategy_config = {
        "asset": coin1,
        "long_only": 0,                  # DMP trades both sides; `side` selects the kernel
        "VT": float(g["VT_vol_target"]),
        "txn_cost": float(g["FIXED_COST_per_trade"]),
        "slippage_per_turn": slippage,
        "fill_window_min": int(g["fill_window_min"]),
        "median_window_bars": int(g["median_window_bars"]),
        "atr_ewm_span": int(g["atr_ewm_span"]),          # 10 — run_cell's override, not prep's 14
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
    sides = tuple(payload["sides"])
    assert sides == DIRECTIONAL_MOMENTUM_SIDES, (sides, DIRECTIONAL_MOMENTUM_SIDES)
    assert stride >= len(sides), f"stride {stride} cannot hold {len(sides)} sides per cell"

    parameter_dict = {}
    for tf_idx, tf in enumerate(tfs):
        p = payload["params_per_TF"][str(tf)]
        annf = math.sqrt(strategy_config["annualization_days_PER"] * 24 * 60 / tf)
        assert abs(annf - float(p["annf"])) < 0.01, (tf, annf, p["annf"])
        for side_idx, side in enumerate(sides):
            ## dde_mult = M_side * DDE_base, where DDE_base follows the frozen law
            ## round(77/sqrt(TF)). params.json stores both the base and the product; assert
            ## the product against the base so a hand-edit of either is caught.
            dde_mult = float(p[f"dde_mult_{side}"])
            assert abs(dde_mult / float(p["DDE_base"]) - (4.0 if side == "LONG" else 1.5)) < 1e-12, \
                (tf, side, dde_mult, p["DDE_base"])
            pid = start_parent_id + stride * tf_idx + side_idx
            parameter_dict[pid] = DIRECTIONAL_MOMENTUM_PARAM_TUPLE(
                parent_stratid=pid,
                coin1=coin1,
                side=side,
                tf=tf,
                agg_time=f"{tf}T",
                K=float(p["K"]),
                X=float(p["X"]),
                Z=float(p["Z"]),
                EZ=float(p["EZ"]),
                TT=float(p["TT"]),
                T1=float(p["T1"]),
                dde_mult=dde_mult,
                annf=annf,
                strategy_config=strategy_config,
                exec_type=None,   # bundle has no execution route; see the field comment
            )

    assert len(parameter_dict) == len(tfs) * len(sides), len(parameter_dict)
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

    Twin of pair_momentum_utils.plain_symbol. Deliberately duplicated rather than
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


def build_directional_momentum_parameter_dict_from_db(db_rows, coin1, sleeve_paths):
    """Build {parent_stratid: DIRECTIONAL_MOMENTUM_PARAM_TUPLE} for ONE asset from its
    `submodel_parameters` rows (is_live = 1) — in LIVE mode the DB is the single source
    of truth for parent ids AND kernel parameters; the frozen bundle is NOT required on
    the live host. Artifact paths ride in via the client config's sleeve_config block.

    db_rows: {parent_trading_model: model_parameters} as dumped by
    crypto-infra/db_scripts (get_dump_dict_directional_momentum_som). The same
    structural laws the bundle builder asserts are re-asserted here from the row values
    (dde_mult = M_side * dde_base, annf = sqrt(PER*24*60/tf)).
    """
    params_path = sleeve_paths["params_path"]
    strategy_config = None
    parameter_dict = {}
    for pid in sorted(db_rows):
        mp = db_rows[pid]
        assert mp["strategy_name"] == "directional_momentum", (pid, mp["strategy_name"])
        assert mp["coin"] == coin1, (pid, mp["coin"], coin1)

        if strategy_config is None:
            strategy_config = {
                "asset": coin1,
                "long_only": 0,                  # DMP trades both sides; `side` selects the kernel
                "VT": float(mp["vt_vol_target"]),
                "txn_cost": float(mp["fixed_cost_per_trade"]),
                "slippage_per_turn": float(mp["slippage_per_turn"]),
                "fill_window_min": int(mp["fill_window_min"]),
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
        m_side = 4.0 if int(mp["long_only"]) == 1 else 1.5
        assert abs(float(mp["dde_mult"]) / float(mp["dde_base"]) - m_side) < 1e-12, \
            (pid, mp["dde_mult"], mp["dde_base"])

        parameter_dict[pid] = DIRECTIONAL_MOMENTUM_PARAM_TUPLE(
            parent_stratid=pid,
            coin1=coin1,
            side="LONG" if int(mp["long_only"]) == 1 else "SHORT",
            tf=tf,
            agg_time=mp["agg_time"],
            K=float(mp["k"]),
            X=float(mp["x"]),
            Z=float(mp["z"]),
            EZ=float(mp["ez"]),
            TT=float(mp["tt"]),
            T1=float(mp["t1"]),
            dde_mult=float(mp["dde_mult"]),
            annf=annf,
            strategy_config=strategy_config,
            ## NOT NULL downstream in live_signals -- fail here, where the offending pid is
            ## named, rather than at the insert.
            exec_type=int(mp["exec_type"]),
        )

    assert len(parameter_dict) == len(db_rows), (len(parameter_dict), len(db_rows))
    return parameter_dict


class DirectionalMomentumBaseClass:
    logger_name = ""
