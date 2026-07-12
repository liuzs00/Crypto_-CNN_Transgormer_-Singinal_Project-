"""
btc_meta_label.py — Meta-labeling (López de Prado) M2 layer on the frozen production model.

M1 (the conv-stem transformer, emb_cross_nf) decides trade DIRECTION. A secondary model M2
(gradient-boosted trees) scores whether each M1 signal will be PROFITABLE, from M1's own
conviction + market-regime context. That score becomes a confidence FILTER and a continuous
position SIZE (higher confidence → larger allocation), suppressing false positives in
unfavourable regimes.

Validated leak-safe (M2 trains on train-split trades, threshold on val, judged on the test
backtest): across 5 M2 seeds, profit factor 1.93 → ~2.9, win-rate 58% → 66%, drawdown
−33% → −25%, all 5/5. See README / commit history.

This module is BOTH the trainer and the serving library — `m2_features()` and
`recommended_size()` are imported by btc_predict_cross_nf.py so train and serve build the
exact same M2 inputs (no parity drift).

Run:  py -3.10 btc_meta_label.py            # 5-seed validation + train & save production M2
"""
import os, sys, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd
import joblib
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
import btc_backtest as B
import btc_backtest_cross_nf as BT

from paths import MODELS_DIR
M2_PATH  = os.path.join(MODELS_DIR, 'btc_eth_sol_meta_m2.pkl')   # saved alongside the M1 .pth
SKIP_THR, FULL_THR = 0.50, 0.70        # size = 0 below SKIP_THR, ramps to full by FULL_THR
MACRO_W  = 64                          # ~11-day (64×4h) rolling realised-vol window
USE_CONVDYN = os.environ.get('CONVDYN', '0') == '1'   # M2 conviction-dynamics — REJECTED walk-forward A/B (exp Δ -0.000%, cell-win 15/30); M1 already saturates its own signal-state info. Kept default-off.

# Curated M2 context: regime / vol / systemic / trend — NOT all 291 features (M2 adds regime
# judgement on top of M1, it must not relearn M1's direction call).
CTX = ['4h_atr14', '4h_rvol20', '4h_bbw', '1d_atr14', 'x_absorb', 'x_absorb_chg',
       '4h_ema50_200', '4h_pr200', 'x_relvol', 'x_corr']

# Conviction-dynamics (experiment): summarise the recent M1 SIGNAL-STATE trajectory — leak-free
# (lagged M1 outputs only, no M2 self-reference), the compact "is conviction stable/rising or a
# flip-flop chop" signal that the +0.43 signal-autocorrelation says may inform trade quality.
CONVDYN_COLS = ['cd_dir_persist6', 'cd_dir_persist18', 'cd_conf_mom6',
                'cd_conf_std6', 'cd_active_dens6', 'cd_agree6']


def macro_vol(close, w=MACRO_W):
    return pd.Series(close).pct_change().rolling(w).std().values


def _conv_dynamics(d, idx, is_long):
    """Causal M1 conviction-dynamics for each entry bar in `idx`: summarise the LAGGED signal-state
    trajectory. Uses only bars strictly BEFORE the entry for the lagged windows (+ the entry's own
    prob_long, which M2 already sees) → leak-free and non-circular (M1 outputs only). Needs
    d['probs'] (per-bar P(long)) and d['sig'] (per-bar LONG/SHORT/NEUTRAL)."""
    probs = np.asarray(d['probs'], float); sig = np.asarray(d['sig'])
    dirn  = np.where(sig == 'LONG', 1.0, np.where(sig == 'SHORT', -1.0, 0.0))
    idx = np.asarray(idx, int); is_long = np.asarray(is_long, bool)
    out = {c: np.zeros(len(idx)) for c in CONVDYN_COLS}
    for k, i in enumerate(idx):
        d6 = dirn[max(0, i-6):i]; d18 = dirn[max(0, i-18):i]
        p6 = probs[max(0, i-6):i]; p6 = p6[~np.isnan(p6)]
        pi = probs[i] if not np.isnan(probs[i]) else 0.5
        cur = 1.0 if is_long[k] else -1.0
        out['cd_dir_persist6'][k]  = d6.mean()  if d6.size  else 0.0     # recent directional bias (streak)
        out['cd_dir_persist18'][k] = d18.mean() if d18.size else 0.0     # longer-horizon bias
        out['cd_conf_mom6'][k]     = pi - (p6.mean() if p6.size else pi) # conviction momentum (current vs recent)
        out['cd_conf_std6'][k]     = p6.std()   if p6.size > 1 else 0.0  # conviction volatility (steady vs flip-flop)
        out['cd_active_dens6'][k]  = (d6 != 0).mean() if d6.size else 0.0# recent active-signal density
        out['cd_agree6'][k]        = (np.sign(d6) == cur).mean() if d6.size else 0.0  # recent agreement w/ this call
    return pd.DataFrame(out)


def m2_features(d, idx, is_long, prob_long, convdyn=None):
    """M2 input matrix for candle indices `idx` (SHARED by trainer + predict serving).
    d must carry d['X'] (scaled features), d['feat_cols'], d['close'] (+ d['probs'], d['sig'] if
    conviction-dynamics is on). `convdyn` defaults to the CONVDYN env flag; pass True/False to
    override (the A/B driver builds both matrices from one process)."""
    X, feat_cols = d['X'], d['feat_cols']
    cols = {c: feat_cols.index(c) for c in CTX if c in feat_cols}
    mv = macro_vol(d['close'])
    idx = np.asarray(idx, dtype=int); is_long = np.asarray(is_long, dtype=bool)
    prob_long = np.asarray(prob_long, dtype=float)
    feats = {'m1_prob': prob_long,
             'm1_conv': np.where(is_long, prob_long, 1 - prob_long),   # conviction in chosen dir
             'm1_dir':  np.where(is_long, 1.0, -1.0)}
    for c, j in cols.items(): feats[c] = X[idx, j]
    feats['macro_vol'] = mv[idx]
    df = pd.DataFrame(feats)
    if USE_CONVDYN if convdyn is None else convdyn:
        df = pd.concat([df, _conv_dynamics(d, idx, is_long)], axis=1)
    return df


def recommended_size(p, skip=SKIP_THR, full=FULL_THR):
    """Continuous position size in [0,1]: 0 below `skip`, ramps linearly to 1 by `full`."""
    return float(np.clip((np.asarray(p, float) - skip) / (full - skip), 0.0, 1.0))


def dataset_from_trades(d, trades, convdyn=None):
    ei = trades['entry_idx'].values.astype(int)
    is_long = (trades['direction'].values == 'LONG')
    Xm = m2_features(d, ei, is_long, trades['prob_long'].values, convdyn=convdyn)
    y = (trades['net'].values > 0).astype(int)             # target: was the M1 trade profitable?
    sp = d['split'][ei]
    return Xm, y, sp


def fit_m2(Xm, y, seed=0):
    """Ensemble M2: soft-vote (mean probability) of GBM (HistGBM ≈ LightGBM) + RandomForest.
    Boosting + bagging diversity — the blend beat GBM-alone in 6/6 walk-forward folds.
    The VotingClassifier exposes predict_proba like a single model, so serving is unchanged."""
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=seed)
    rf  = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=50,
                                 max_features='sqrt', random_state=seed, n_jobs=-1)
    ens = VotingClassifier([('gbm', gbm), ('rf', rf)], voting='soft')
    ens.fit(Xm, y); return ens


def load_m2(path=M2_PATH):
    return joblib.load(path)        # dict: m2, feat_order, skip_thr, full_thr, ctx, macro_w


def pf_of(net):
    gp = net[net > 0].sum(); gl = -net[net < 0].sum()
    return gp / gl if gl > 0 else float('inf')


def _show(tr_sub, d, label):
    eq, used = B.equity_curve(tr_sub, d)
    if len(eq) < 2:
        print(f"  {label:26s} no curve"); return None
    ret = (eq['equity'].iloc[-1]-1)*100; mdd = B.max_drawdown(eq['equity'].values)*100
    net = used['net'].values
    print(f"  {label:26s} trades:{len(used):4d}  win%:{(net>0).mean()*100:5.1f}  "
          f"PF:{pf_of(net):4.2f}  return:{ret:+8.1f}%  maxDD:{mdd:6.1f}%")
    return dict(pf=pf_of(net), ret=ret, mdd=mdd)


def main():
    print("M1: production emb_cross_nf (frozen)   ·   M2: HistGradientBoosting meta-label")
    d = BT.build()
    trades = B.run_trades(d, 'full')
    Xm, y, sp = dataset_from_trades(d, trades)
    tr, va, te = sp == 'train', sp == 'val', sp == 'test'
    net_va = trades['net'].values[va]; te_trades = trades[te].copy()
    print(f"\n  M1 active signals — train:{tr.sum()}  val:{va.sum()}  test:{te.sum()}   "
          f"(profitable-rate: train {y[tr].mean():.3f}  test {y[te].mean():.3f})")

    # ── leak-safe validation: 5 M2 seeds, train→test ──
    print("\n=== VALIDATION: M1-alone vs M1+M2 filter (test, 5 M2 seeds) ===")
    base = _show(te_trades, d, "M1 alone (all signals)")
    pf05, pfvt, aucs = [], [], []
    for seed in range(5):
        m2 = fit_m2(Xm[tr], y[tr], seed)
        p_va = m2.predict_proba(Xm[va])[:, 1]; p_te = m2.predict_proba(Xm[te])[:, 1]
        try: aucs.append(roc_auc_score(y[te], p_te))
        except Exception: pass
        best_thr, best = 0.50, -1
        for thr in np.round(np.arange(0.30, 0.71, 0.02), 2):
            sel = p_va > thr
            if sel.sum() >= 25 and pf_of(net_va[sel]) > best: best, best_thr = pf_of(net_va[sel]), thr
        r05 = _show(te_trades[p_te > 0.50], d, f"seed{seed} thr0.50")
        rvt = _show(te_trades[p_te > best_thr], d, f"seed{seed} thr{best_thr}(val)")
        if r05: pf05.append(r05['pf'])
        if rvt: pfvt.append(rvt['pf'])
    print(f"\n  M2 test AUC mean {np.mean(aucs):.3f} (all>0.5: {all(a>0.5 for a in aucs)})  |  "
          f"PF thr0.50 {np.mean(pf05):.2f} ({sum(p>base['pf'] for p in pf05)}/5)  "
          f"PF val-thr {np.mean(pfvt):.2f} ({sum(p>base['pf'] for p in pfvt)}/5)  vs M1 {base['pf']:.2f}")

    # ── train the PRODUCTION M2 on ALL resolved trades (full history), save next to M1 ──
    m2 = fit_m2(Xm, y, seed=0)
    joblib.dump({'m2': m2, 'feat_order': list(Xm.columns), 'ctx': CTX, 'convdyn': USE_CONVDYN,
                 'skip_thr': SKIP_THR, 'full_thr': FULL_THR, 'macro_w': MACRO_W}, M2_PATH)
    print(f"\nSaved production M2 → {os.path.basename(M2_PATH)}  "
          f"(GBM+RF ensemble on {len(y):,} resolved M1 trades; size ramp {SKIP_THR}->{FULL_THR})")


if __name__ == '__main__':
    main()
