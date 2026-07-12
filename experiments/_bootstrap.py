"""
_bootstrap.py — put `src/` on the import path.

Experiment scripts live outside `src/`, so Python's sys.path[0] is `experiments/` and the
production modules (crypto_features, btc_backtest, btc_meta_label, paths, …) aren't importable.
Import this FIRST in any experiment script:

    import _bootstrap  # noqa: F401  — puts src/ on sys.path
    import btc_meta_label as ML
    from paths import DATA_DIR, MODELS_DIR
"""
import os, sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
