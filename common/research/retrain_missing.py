#!/usr/bin/env python3
"""Quick retrain for specific symbols — loads Optuna params from SQLite DB."""
import sys, os, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config, cfg
from engine.inference import SovereignMLFilter
from audit.audit_logger import BlackoutLogger

load_config()

# Merge BF symbols into cfg.SYMBOLS
bf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "sovereign_configs_bf.json")
bf_cfg = json.load(open(bf_path))
for sym, sym_cfg in bf_cfg.items():
    if sym == "margin_leverage":
        continue
    if sym not in cfg.SYMBOLS:
        cfg.SYMBOLS[sym] = sym_cfg

# Load standard CSV params first
cfg.load_optuna_params()

# Then load from BF Optuna SQLite DB (overrides CSV if present)
bf_db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "optuna_bf_H4_54sym", "optuna_studies.db")
if os.path.exists(bf_db):
    conn = sqlite3.connect(bf_db)
    cursor = conn.execute("""
        SELECT s.study_name, t.number, tv.value
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY s.study_name, tv.value DESC
    """)
    # Group by study to find best trial per symbol
    best_trials = {}
    for study_name, trial_number, value in cursor:
        # Strip _H4/_H1 timeframe suffix, then normalize underscores
        sym = study_name
        for suffix in ("_H4", "_H1", "_M15", "_D1"):
            if sym.endswith(suffix):
                sym = sym[:-len(suffix)]
                break
        # Normalize: ALGO_USD → ALGOUSD, ADA_USD → ADAUSD
        sym = sym.replace("_", "").replace("/", "")
        if sym not in best_trials or value > best_trials[sym][1]:
            best_trials[sym] = (trial_number, value, study_name)

    for sym, (trial_num, best_ev, study_name) in best_trials.items():
        if sym in cfg.OPTUNA_PARAMS:
            continue  # Already has params from CSV
        # Load trial params
        study_id = conn.execute(
            "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
        ).fetchone()[0]
        params = {}
        for name, value in conn.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ("
            "  SELECT trial_id FROM trials WHERE study_id = ? AND number = ?"
            ")", (study_id, trial_num)):
            params[name] = float(value)

        if not params:
            continue

        cfg.OPTUNA_PARAMS[sym] = {
            "xgb_params": {
                "booster": "gbtree",
                "tree_method": "hist",
                "device": "cuda",
                "sampling_method": "gradient_based",
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": int(params.get("max_depth", 6)),
                "eta": params.get("eta", 0.03),
                "gamma": params.get("gamma", 0.1),
                "subsample": params.get("subsample", 0.7),
                "colsample_bytree": params.get("colsample_bytree", 0.7),
                "colsample_bylevel": params.get("colsample_bylevel", 0.75),
                "reg_alpha": params.get("reg_alpha", 0.1),
                "reg_lambda": params.get("reg_lambda", 1.0),
                "min_child_weight": params.get("min_child_weight", 5.0),
                "max_bin": 512,
                "grow_policy": "lossguide",
                "verbosity": 0,
            },
            "num_boost_round": int(params.get("num_boost_round", 500)),
            "optuna_ev": best_ev,
            "timeframe": "H4",
        }
        print(f"  [DB] Loaded params for {sym} (EV={best_ev:.6f}, trial #{trial_num})")
    conn.close()

logger = BlackoutLogger()

# Symbols to train (from CLI or hardcoded missing list)
if len(sys.argv) > 1:
    symbols = sys.argv[1:]
else:
    symbols = ["ALGOUSD", "AUDCAD", "AUDCHF", "EURCAD", "EURCHF",
               "EURGBP", "GBPCHF", "NZDCAD", "NZDCHF", "NZDJPY"]

print(f"\nTraining {len(symbols)} symbols: {symbols}")
trained = 0
for sym in symbols:
    if sym not in cfg.OPTUNA_PARAMS:
        print(f"\n  {sym}: no Optuna params found, skipping", flush=True)
        continue
    print(f"\n  Training {sym}...", flush=True)
    filt = SovereignMLFilter(sym, logger)
    try:
        if filt.train_model():
            trained += 1
            print(f"    OK", flush=True)
        else:
            print(f"    SKIPPED (no data)", flush=True)
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)

print(f"\nDone: {trained}/{len(symbols)} trained.")
