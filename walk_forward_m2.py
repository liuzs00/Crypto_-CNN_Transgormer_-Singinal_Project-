"""
walk_forward_m2.py — Expanding-window walk-forward validation of the M2 meta-label.

The single train/val/test split answers "does M2 help on ONE held-out window". This answers
the harder, honest question the PSI feature-drift flagged: does M2's benefit SURVIVE regime
drift across MANY sequential out-of-sample windows?

For each expanding fold: fit M2 on the window's past, tune the filter threshold on a recent
tail of that past (leak-safe), then test on the window's future. Average across folds and 5
M2 seeds. Compares the default M2 config vs a more-regularized one for drift-robustness.

(Scratch experiment — not wired into production.)
"""
import sys, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
import btc_backtest as B
import btc_backtest_cross_nf as BT
import btc_meta_label as ML

CONFIGS = {
    'default':     dict(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0, min_samples_leaf=20),
    'regularized': dict(max_depth=3, learning_rate=0.03, max_iter=600, l2_regularization=2.0, min_samples_leaf=100),
}
N_FOLDS   = 6
INIT_FRAC = 0.40     # initial expanding-window training base
INNER_VAL = 0.15     # recent tail of past used to tune the filter threshold (leak-safe)
SEEDS     = range(5)


def fit(Xm, y, cfg, seed):
    m = HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.15,
                                       random_state=seed, **cfg)
    m.fit(Xm, y); return m


def bt(trades, d):
    eq, used = B.equity_curve(trades, d)
    if len(eq) < 2: return None
    net = used['net'].values
    return dict(n=len(used), pf=ML.pf_of(net), exp=net.mean() * 100, mdd=B.max_drawdown(eq['equity'].values) * 100)


def tune_thr(p, net):
    best_thr, best = 0.50, -1e9
    for thr in np.round(np.arange(0.30, 0.81, 0.02), 2):
        sel = p > thr
        if sel.sum() >= 20:
            pf = ML.pf_of(net[sel])
            if pf > best: best, best_thr = pf, thr
    return best_thr


def main():
    d = BT.build()
    trades = B.run_trades(d, 'full').sort_values('entry_idx').reset_index(drop=True)
    Xm, y, _ = ML.dataset_from_trades(d, trades)
    net_all = trades['net'].values; N = len(trades)
    init = int(N * INIT_FRAC); region = N - init
    bounds = [init + int(region * k / N_FOLDS) for k in range(N_FOLDS + 1)]
    print(f"  trades:{N:,}  initial train:{init:,}  {N_FOLDS} expanding folds of ~{region // N_FOLDS} trades each")

    for cname, cfg in CONFIGS.items():
        print(f"\n=== config: {cname} ===")
        print(f"  {'fold':>4} {'M1 PF':>7} {'M2 PF':>7} {'ΔPF':>6} {'M1 exp':>7} {'M2 exp':>7} {'M2 n':>6}")
        m1_pfs, m2_pfs, m1_exps, m2_exps, pf_wins, exp_wins = [], [], [], [], 0, 0
        for k in range(N_FOLDS):
            f0, f1 = bounds[k], bounds[k + 1]
            past = np.arange(0, f0); fut = np.arange(f0, f1)
            vcut = int(len(past) * (1 - INNER_VAL)); fit_idx, inval = past[:vcut], past[vcut:]
            r1, r2, nn = [], [], []
            e1, e2 = [], []
            for seed in SEEDS:
                m = fit(Xm.iloc[fit_idx], y[fit_idx], cfg, seed)
                thr = tune_thr(m.predict_proba(Xm.iloc[inval])[:, 1], net_all[inval])
                p_fut = m.predict_proba(Xm.iloc[fut])[:, 1]
                ft = trades.iloc[fut]
                a = bt(ft, d); b = bt(ft.iloc[np.where(p_fut > thr)[0]], d)
                if a and b:
                    r1.append(a['pf']); r2.append(b['pf']); nn.append(b['n']); e1.append(a['exp']); e2.append(b['exp'])
            m1pf, m2pf = np.mean(r1), np.mean(r2); m1e, m2e = np.mean(e1), np.mean(e2)
            m1_pfs.append(m1pf); m2_pfs.append(m2pf); m1_exps.append(m1e); m2_exps.append(m2e)
            pf_wins += (m2pf > m1pf); exp_wins += (m2e > m1e)
            print(f"  {k+1:>4} {m1pf:7.2f} {m2pf:7.2f} {m2pf-m1pf:+6.2f} {m1e:+6.2f}% {m2e:+6.2f}% {np.mean(nn):6.0f}")
        print(f"  SUMMARY  PF {np.mean(m1_pfs):.2f}→{np.mean(m2_pfs):.2f} (M2 wins {pf_wins}/{N_FOLDS})   "
              f"exp/trade {np.mean(m1_exps):+.2f}%→{np.mean(m2_exps):+.2f}% (M2 wins {exp_wins}/{N_FOLDS})")


if __name__ == '__main__':
    main()
