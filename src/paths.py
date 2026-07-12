"""
paths.py — single source of truth for every filesystem location in the project.

ROOT is derived from this file's own location (`src/paths.py` → parent of `src/`), so the project
is PORTABLE: clone it anywhere, on any machine, and every script resolves its data/model/output
paths correctly. Previously each script hardcoded an absolute `D:\\Document\\...` path (19 of them),
which meant the repo only ran on one machine at one exact location.

Layout:
    ROOT/
      DATA/        raw market data (Binance OHLCV, funding/perp, NQ/ES macro, gold)
      src/         production pipeline (this package)
      experiments/ research / tuning scripts (not production)
      models/      trained checkpoints (*.pth, *.pkl)
      outputs/     signals, trades, plots
      logs/        run logs + the append-only signal-state history
      scripts/     run_predict.bat, register_predict_task.ps1
      config/      telegram_config.json (gitignored) + .example
      tests/       tests
      docs/        FEATURES.md, TECHNICAL_REPORT.md

Usage:
    from paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, CONFIG_DIR, SRC_DIR
Scripts outside `src/` (e.g. experiments/) must first put `src/` on the import path:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
"""
import os

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(SRC_DIR)

DATA_DIR    = os.path.join(ROOT, 'DATA')
MODELS_DIR  = os.path.join(ROOT, 'models')
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')
LOGS_DIR    = os.path.join(ROOT, 'logs')
CONFIG_DIR  = os.path.join(ROOT, 'config')
EXP_DIR     = os.path.join(ROOT, 'experiments')

# create the writable dirs on import so a fresh clone works with no manual setup
for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, CONFIG_DIR):
    os.makedirs(_d, exist_ok=True)
