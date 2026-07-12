"""
m2_hp_search.py — Hyper-parameter / sizing / calibration search for the M2 meta-label.

Budget focused on min_samples_leaf × learning_rate (+ max_depth, l2). Two-stage, leak-safe:

  Stage 1 (screen)  : tree-HP grid scored on the held-out VAL split with a THRESHOLD-FREE
                      economic objective (expectancy of the top-COVER% by M2 confidence) so
                      configs are ranked fairly without threshold-fishing. 3 M2 seeds.
  Stage 2 (confirm) : the Stage-1 winner vs the current default on the EXPANDING WALK-FORWARD
                      (6 folds × 5 seeds) — the drift-robust metric. This is the decision.
  Stage 3 (axes)    : calibration (val ECE) and sizing-curve (continuous size vs binary filter).

High adoption bar: adopt only if the winner beats the default on the walk-forward mean
expectancy AND wins the majority of folds — otherwise confirm the default.

(Scratch experiment — not wired into production.)
"""
import sys, warnings, itertools
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
import _bootstrap  # noqa: F401  — puts src/ on sys.path
import btc_backtest as B
import btc_backtest_cross_nf as BT
import btc_meta_label as ML
from crypto_metrics import calibration

COVER = 0.60
GRID = dict(min_samples_leaf=[20, 50, 100, 200], learning_rate=[0.02, 0.05, 0.1],
            max_depth=[3, 4], l2_regularization=[0.5, 2.0])
DEFAULT = dict(min_samples_leaf=20, learning_rate=0.05, max_depth=3, l2_regularization=1.0)
N_FOLDS, INIT_FRAC, INNER_VAL = 6, 0.40, 0.15


def fit(Xm, y, cfg, seed):
    return HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                          validation_fraction=0.15, random_state=seed, **cfg).fit(Xm, y)


def econ_topk(p, net, cover=COVER):
    k = max(int(len(p) * cover), 1); idx = np.argsort(p)[::-1][:k]; s = net[idx]
    return s.mean(), ML.pf_of(s)


def tune_thr(p, net):
    bt, bv = 0.50, -1e9
    for t in np.round(np.arange(0.30, 0.81, 0.02), 2):
        m = p > t
        if m.sum() >= 20 and ML.pf_of(net[m]) > bv: bv, bt = ML.pf_of(net[m]), t
    return bt


def walk_forward(Xm, y, net, trades, d, cfg, seeds=5):
    """Mean per-trade expectancy of M1-alone vs M1+M2 across expanding folds (avg over seeds)."""
    N = len(trades); init = int(N * INIT_FRAC); region = N - init
    bounds = [init + int(region * k / N_FOLDS) for k in range(N_FOLDS + 1)]
    fold_m1, fold_m2 = [], []
    for k in range(N_FOLDS):
        f0, f1 = bounds[k], bounds[k + 1]; past = np.arange(0, f0); fut = np.arange(f0, f1)
        vcut = int(len(past) * (1 - INNER_VAL)); fi, iv = past[:vcut], past[vcut:]
        e1s, e2s = [], []
        for s in range(seeds):
            m = fit(Xm.iloc[fi], y[fi], cfg, s)
            thr = tune_thr(m.predict_proba(Xm.iloc[iv])[:, 1], net[iv])
            p = m.predict_proba(Xm.iloc[fut])[:, 1]
            ft = trades.iloc[fut]
            a = B.equity_curve(ft, d)[1]; b = B.equity_curve(ft.iloc[np.where(p > thr)[0]], d)[1]
            if len(a) and len(b): e1s.append(a['net'].mean() * 100); e2s.append(b['net'].mean() * 100)
        fold_m1.append(np.mean(e1s)); fold_m2.append(np.mean(e2s))
    wins = sum(b > a for a, b in zip(fold_m1, fold_m2))
    return np.mean(fold_m1), np.mean(fold_m2), wins


def main():
    d = BT.build()
    trades = B.run_trades(d, 'full').sort_values('entry_idx').reset_index(drop=True)
    Xm, y, sp = ML.dataset_from_trades(d, trades)
    tr, va = sp == 'train', sp == 'val'; net = trades['net'].values

    # ── Stage 1: tree-HP grid (val, threshold-free economics) ──
    print(f"=== Stage 1: tree-HP grid — val top-{int(COVER*100)}% economics, 3 seeds, {np.prod([len(v) for v in GRID.values()])} configs ===")
    keys = list(GRID); rows = []
    for vals in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, vals)); es = []
        for s in range(3):
            m = fit(Xm[tr], y[tr], cfg, s); e, _ = econ_topk(m.predict_proba(Xm[va])[:, 1], net[va]); es.append(e)
        rows.append((cfg, np.mean(es)))
    rows.sort(key=lambda r: -r[1])
    de = np.mean([econ_topk(fit(Xm[tr], y[tr], DEFAULT, s).predict_proba(Xm[va])[:, 1], net[va])[0] for s in range(3)])
    print(f"  default config exp/trade: {de*100:+.3f}%")
    print("  top 6 by val expectancy:")
    for cfg, e in rows[:6]:
        flag = '  <-- = default' if cfg == DEFAULT else ''
        print(f"    {e*100:+.3f}%   {cfg}{flag}")
    best = rows[0][0]

    # ── Stage 3 (axes): calibration + sizing on val (default config, seed 0) ──
    m0 = fit(Xm[tr], y[tr], DEFAULT, 0); pv = m0.predict_proba(Xm[va])[:, 1]; yv = y[va]
    brier, ece = calibration(pv, yv)
    # sizing: continuous size-weighted expectancy vs binary top-60%
    sizes = np.clip((pv - 0.50) / (0.70 - 0.50), 0, 1)
    sized_exp = (sizes * net[va]).sum() / (sizes.sum() + 1e-9) * 100
    filt_exp = econ_topk(pv, net[va])[0] * 100
    print(f"\n=== Stage 3: axes (val, default cfg) ===")
    print(f"  Calibration   Brier:{brier:.4f}  ECE:{ece:.4f}")
    print(f"  Sizing        continuous-size exp/trade {sized_exp:+.3f}%   vs   binary top-60% {filt_exp:+.3f}%")

    # ── Stage 2: walk-forward confirmation — best vs default ──
    print(f"\n=== Stage 2: WALK-FORWARD confirmation (6 folds × 5 seeds) ===")
    for name, cfg in [('default', DEFAULT), ('grid-winner', best)]:
        m1e, m2e, wins = walk_forward(Xm, y, net, trades, d, cfg)
        print(f"  {name:12s} {cfg}")
        print(f"               M1 exp {m1e:+.3f}% → M2 exp {m2e:+.3f}%  (M2 wins {wins}/{N_FOLDS} folds)")
    print("\n  ADOPTION BAR: grid-winner adopts only if its walk-forward M2 expectancy beats the")
    print("  default's by > seed noise AND it's not the default already. Otherwise: keep default.")


if __name__ == '__main__':
    main()
