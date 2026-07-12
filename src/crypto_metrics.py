"""
crypto_metrics.py — Shared evaluation, regime thresholding, and attention analysis.

All training scripts call these so the metric/threshold logic lives in one place (the same
regime gates are mirrored in btc_predict.py / btc_backtest for serving).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── volatility-regime-adaptive confidence gates ──
VOL_FEATURE     = '4h_atr14'
LONG_THRESH     = 0.60
SHORT_THRESH    = 0.40
THRESH_LOW_VOL  = (0.55, 0.45)
THRESH_HIGH_VOL = (0.72, 0.28)


def regime_signals(p_long, vol, p33, p67):
    """Map P(long) + scaled-ATR vol → (signal, regime, long_thr, short_thr).
    signal: 1=LONG, 0=SHORT, -1=NEUTRAL."""
    long_thr  = np.where(vol > p67, THRESH_HIGH_VOL[0], np.where(vol < p33, THRESH_LOW_VOL[0], LONG_THRESH))
    short_thr = np.where(vol > p67, THRESH_HIGH_VOL[1], np.where(vol < p33, THRESH_LOW_VOL[1], SHORT_THRESH))
    regime    = np.where(vol > p67, 'high', np.where(vol < p33, 'low', 'mid'))
    sig       = np.where(p_long >= long_thr, 1, np.where(p_long <= short_thr, 0, -1))
    return sig, regime, long_thr, short_thr


def evaluate(model, test_dl, Xte, vol_idx, atr_tr_btc, device, tag,
             feat_cols=None, X_train=None, price_path=None):
    """Run the BTC test set, print argmax + confidence-filtered reports + extended metrics A–E.
    Returns dict with sig, regime, p_long, tprobs, ttrue, p33, p67.
    feat_cols + X_train (train sequences) enable Metric D (PSI); price_path enables Metric E (MAE)."""
    model.eval()
    tpreds, tprobs, ttrue = [], [], []
    import torch
    with torch.no_grad():
        for xb, yb, ab in test_dl:
            out = model(xb.to(device), ab.to(device)); prbs = torch.softmax(out, 1).cpu().numpy()
            tpreds.extend(out.argmax(1).cpu().tolist()); tprobs.extend(prbs.tolist()); ttrue.extend(yb.tolist())
    tpreds = np.array(tpreds); tprobs = np.array(tprobs); ttrue = np.array(ttrue); p_long = tprobs[:, 1]
    p33, p67 = np.percentile(atr_tr_btc, [33, 67])
    vol_te = Xte[:len(tpreds), -1, vol_idx]
    sig, regime, _, _ = regime_signals(p_long, vol_te, p33, p67)
    mask = sig != -1
    print("\n" + "="*55); print(f"BTC+ETH+SOL POOLED ({tag}) — BTC TEST (argmax)"); print("="*55)
    print(f"Accuracy: {accuracy_score(ttrue, tpreds):.4f}")
    print(classification_report(ttrue, tpreds, target_names=['Short', 'Long']))
    print("="*55); print("ADAPTIVE CONFIDENCE-FILTERED SIGNALS (BTC test)"); print("="*55)
    print(f"  LONG:{(sig==1).sum():,}  SHORT:{(sig==0).sum():,}  NEUTRAL:{(sig==-1).sum():,} "
          f"({100*(sig==-1).sum()/len(sig):.1f}% filtered)")
    if mask.sum() > 0:
        print(f"\nAccuracy on active signals: {accuracy_score(ttrue[mask], sig[mask]):.4f}")
        print(classification_report(ttrue[mask], sig[mask], target_names=['Short', 'Long']))
        print("Confusion matrix:"); print(confusion_matrix(ttrue[mask], sig[mask]))
    print(f"\nProb(Long) — mean:{p_long.mean():.3f}  std:{p_long.std():.3f}  "
          f"min:{p_long.min():.3f}  max:{p_long.max():.3f}")
    extended_report(p_long, ttrue, sig, feat_cols=feat_cols,
                    X_train=(X_train[:, -1, :] if X_train is not None else None),
                    X_test=(Xte[:len(ttrue), -1, :] if X_train is not None else None),
                    price_path=price_path)
    return dict(sig=sig, regime=regime, p_long=p_long, tprobs=tprobs, ttrue=ttrue, p33=float(p33), p67=float(p67))


# ══════════════════════════════════════════════════════════════════
# EXTENDED METRICS  (A–E)
# ══════════════════════════════════════════════════════════════════
def calibration(p_long, y_true, n_bins=10):
    """Metric A — Brier score + Expected Calibration Error for P(long). Lower = better."""
    p = np.asarray(p_long, float); y = np.asarray(y_true, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; N = max(len(p), 1)
    for b in range(n_bins):
        hi = edges[b + 1] + (1e-9 if b == n_bins - 1 else 0)
        m = (p >= edges[b]) & (p < hi)
        if m.sum() == 0: continue
        ece += (m.sum() / N) * abs(p[m].mean() - y[m].mean())   # |confidence − accuracy|
    return brier, float(ece)


def class_profit_factor(sig, y_true, ltp=0.03, lsl=0.015, stp=0.03, ssl=0.015):
    """Metric B — barrier-implied profit factor + expectancy per signal class (gross of fees).
    sig codes: 1=LONG, 0=SHORT, -1=NEUTRAL. y_true: 1=long-wins, 0=short-wins."""
    out = {}
    for name, code, win, tp, sl in [('Long', 1, 1, ltp, lsl), ('Short', 0, 0, stp, ssl)]:
        m = sig == code
        if m.sum() == 0: out[name] = None; continue
        correct = (y_true[m] == win)
        g = correct.sum() * tp; l = (~correct).sum() * sl
        out[name] = dict(n=int(m.sum()), win=float(correct.mean()),
                         pf=(g / l if l > 0 else float('inf')), exp=float((g - l) / m.sum()))
    return out


def signal_clustering(sig):
    """Metric C — autocorrelation / clustering of the signal stream (>0 = signals persist/cluster).
    sig codes: 1=LONG, 0=SHORT, -1=NEUTRAL."""
    s = np.where(sig == 1, 1.0, np.where(sig == 0, -1.0, 0.0))   # signed direction
    a = (sig != -1).astype(float)                                # active indicator
    def ac(x, k):
        if len(x) <= k + 1 or x.std() == 0: return float('nan')
        return float(np.corrcoef(x[:-k], x[k:])[0, 1])
    return dict(dir_ac1=ac(s, 1), dir_ac6=ac(s, 6), active_ac1=ac(a, 1))


def psi(X_ref, X_cur, feat_cols, n_bins=10):
    """Metric D — Population Stability Index per feature (reference=train vs current=test).
    <0.1 stable · 0.1–0.25 minor shift · >0.25 major shift. Returns {feature: psi}."""
    vals = {}
    for j, name in enumerate(feat_cols):
        ref = X_ref[:, j]; cur = X_cur[:, j]
        q = np.quantile(ref, np.linspace(0, 1, n_bins + 1)); q[0] = -np.inf; q[-1] = np.inf
        rh = np.histogram(ref, q)[0] / max(len(ref), 1) + 1e-6
        ch = np.histogram(cur, q)[0] / max(len(cur), 1) + 1e-6
        vals[name] = float(np.sum((ch - rh) * np.log(ch / rh)))
    return vals


def max_adverse_excursion(close, high, low, entries, dirs, timeout):
    """Metric E — Maximum Adverse Excursion per trade: worst adverse move (fraction of entry
    price) before the time barrier. entries: candle indices; dirs: 'LONG'/'SHORT'."""
    n = len(close); out = []
    for i, dr in zip(entries, dirs):
        i = int(i); ref = close[i]; e = min(i + 1 + timeout, n)
        seg_lo = low[i + 1:e]; seg_hi = high[i + 1:e]
        if len(seg_lo) == 0 or ref <= 0: out.append(0.0); continue
        adv = (ref - seg_lo.min()) / ref if dr == 'LONG' else (seg_hi.max() - ref) / ref
        out.append(max(float(adv), 0.0))
    return np.array(out)


def extended_report(p_long, y_true, sig, *, feat_cols=None, X_train=None, X_test=None,
                    price_path=None, barriers=(0.03, 0.015, 0.03, 0.015)):
    """Print extended metrics A–E. D (PSI) runs if X_train/X_test/feat_cols given; E (MAE) if
    price_path=(close, high, low, entries, timeout) given."""
    print("\n" + "=" * 55); print("EXTENDED METRICS (A–E)"); print("=" * 55)
    brier, ece = calibration(p_long, y_true)
    print(f"  A. Calibration    Brier:{brier:.4f}   ECE:{ece:.4f}   (lower = better calibrated)")
    print(f"  B. Per-class profit factor / expectancy (barrier-implied, gross):")
    for k, v in class_profit_factor(sig, y_true, *barriers).items():
        if v: print(f"       {k:5s} n={v['n']:4d}  win={v['win']*100:4.1f}%  PF={v['pf']:4.2f}  exp/sig={v['exp']*100:+5.2f}%")
    sc = signal_clustering(sig)
    print(f"  C. Signal autocorr  dir_ac1:{sc['dir_ac1']:+.3f}  dir_ac6:{sc['dir_ac6']:+.3f}  "
          f"active_ac1:{sc['active_ac1']:+.3f}   (>0 = clustered/persistent)")
    if X_train is not None and X_test is not None and feat_cols is not None:
        ps = psi(X_train, X_test, feat_cols); arr = np.array(list(ps.values())); names = list(ps)
        order = np.argsort(arr)[::-1]
        print(f"  D. PSI train→test  mean:{arr.mean():.3f}  max:{arr.max():.3f}  "
              f"(major>0.25:{int((arr>0.25).sum())}  minor:{int(((arr>0.1)&(arr<=0.25)).sum())} of {len(arr)})")
        print("       top drift: " + ", ".join(f"{names[i]}={arr[i]:.2f}" for i in order[:5]))
    if price_path is not None:
        close, high, low, entries, timeout = price_path
        m = sig != -1
        if m.sum() and len(entries) == len(sig):
            dirs = np.where(sig[m] == 1, 'LONG', 'SHORT')
            mae = max_adverse_excursion(close, high, low, np.asarray(entries)[m], dirs, timeout)
            print(f"  E. Max Adverse Excursion (active trades)  mean:{mae.mean()*100:.2f}%  "
                  f"median:{np.median(mae)*100:.2f}%  p90:{np.percentile(mae,90)*100:.2f}%")


def attention_analysis(model, test_dl, Xte, feat_cols, sig, ttrue, device, tag, save_dir,
                       seq_len, patch_size, n_layers):
    """Print + plot received-attention-per-patch / recency / per-layer-entropy / OFI-divergence."""
    import torch
    print("\n" + "="*55); print("ATTENTION MAP ANALYSIS"); print("="*55)
    model.eval()
    n_patches = seq_len // patch_size
    attn_store = []
    with torch.no_grad():
        for xb, _, ab in test_dl:
            _, layer_attns = model(xb.to(device), ab.to(device), return_attn=True)
            attn_store.append(torch.stack(layer_attns, dim=1).cpu().numpy())   # (B, Nl, H, Np, Np)
    attn_all = np.concatenate(attn_store, axis=0)
    recv = attn_all.sum(axis=-2).mean(axis=(1, 2))                              # (N_test, Np)
    patch_pos = np.arange(n_patches)

    print(f"\nReceived attention per patch  (uniform baseline = 1.000):")
    print(f"  {'p':>2}  [steps]  recv    bar")
    for p in range(n_patches):
        val = recv[:, p].mean(); bar = '█' * int(val * 16)
        t = '  ← most recent' if p == n_patches - 1 else ''
        print(f"  {p:2d}  [{p*patch_size:2d}–{p*patch_size+patch_size-1:2d}]  {val:.3f}  {bar}{t}")
    early = recv[:, :4].mean(); late = recv[:, -4:].mean()
    print(f"\nEarly patches [0–3]   mean received: {early:.4f}")
    print(f"Late  patches [12–15] mean received: {late:.4f}")
    print(f"Recency ratio (late / early)        : {late/early:.2f}×")

    long_m = sig == 1; short_m = sig == 0
    if long_m.sum() and short_m.sum():
        print(f"\nRecent-patch attention (last 4 patches):")
        print(f"  Long  signals (n={long_m.sum():,}): {recv[long_m, -4:].mean():.4f}")
        print(f"  Short signals (n={short_m.sum():,}): {recv[short_m, -4:].mean():.4f}")
        print(f"  Early-patch attention (first 4 patches):")
        print(f"  Long  signals: {recv[long_m, :4].mean():.4f}")
        print(f"  Short signals: {recv[short_m, :4].mean():.4f}")
        if recv[short_m, :4].mean() > recv[long_m, :4].mean():
            print("  → Short signals draw relatively more attention from older patches")

    print(f"\nPer-layer attention entropy  (lower = more focused):")
    for l in range(n_layers):
        a = attn_all[:, l].mean(axis=(0, 1)); ent = -(a * np.log(a + 1e-9)).sum(axis=-1).mean()
        print(f"  Layer {l+1}: {ent:.3f}  {'░' * int(ent * 8)}")

    ofi_idx = feat_cols.index('4h_ofi_div') if '4h_ofi_div' in feat_cols else None
    if ofi_idx is not None:
        ofi_vals = Xte[:len(sig), -1, ofi_idx]; recent_attn = recv[:, -4:].mean(axis=1)
        print(f"\nOFI-divergence vs recent-patch attention:")
        for lbl, m in [('Long', long_m), ('Short', short_m)]:
            if m.sum() > 5:
                r = np.corrcoef(np.abs(ofi_vals[m]), recent_attn[m])[0, 1]
                print(f"  {lbl:5s} (n={m.sum():,}): r = {r:+.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for lbl, dat, col, ls in [('All test', recv, 'gray', '-'),
                              ('Long (active)', recv[long_m], 'steelblue', '-'),
                              ('Short (active)', recv[short_m], 'tomato', '-'),
                              ('Correct', recv[(sig != -1) & (sig == ttrue)], 'green', '--'),
                              ('Wrong', recv[(sig != -1) & (sig != ttrue)], 'orange', '--')]:
        if len(dat): ax.plot(patch_pos, dat.mean(0), marker='o', markersize=4, label=f'{lbl} (n={len(dat)})', color=col, linestyle=ls)
    ax.axvspan(11.5, 15.5, alpha=0.08, color='green'); ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, label='uniform')
    ax.set_xlabel('Patch index (0=oldest · 15=most recent)'); ax.set_ylabel('Mean received attention')
    ax.set_title('Received attention per patch — by signal type'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    heat = attn_all[:, -1].mean(axis=(0, 1)); im = ax.imshow(heat, cmap='plasma', aspect='auto', origin='upper')
    ax.set_xlabel('Key patch'); ax.set_ylabel('Query patch'); ax.set_title(f'Layer {n_layers} mean attention map'); plt.colorbar(im, ax=ax)
    ax = axes[1, 0]
    for l in range(n_layers):
        ax.plot(patch_pos, attn_all[:, l].sum(axis=-2).mean(axis=(0, 1)), marker='o', markersize=3, label=f'Layer {l+1}')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Patch index'); ax.set_ylabel('Mean received attention'); ax.set_title('Received attention by layer'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    for lbl, m, col, mk in [('Long correct', (sig==1)&(sig==ttrue), 'steelblue', 'o'),
                            ('Long wrong', (sig==1)&(sig!=ttrue), 'lightblue', 's'),
                            ('Short correct', (sig==0)&(sig==ttrue), 'tomato', 'o'),
                            ('Short wrong', (sig==0)&(sig!=ttrue), 'salmon', 's')]:
        if m.sum() > 0: ax.plot(patch_pos, recv[m].mean(0), marker=mk, markersize=4, label=f'{lbl} n={m.sum()}', color=col)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Patch index'); ax.set_ylabel('Mean received attention'); ax.set_title('Attention — correct vs wrong'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle(f'BTC+ETH+SOL ({tag}) — Attention Map Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    attn_path = os.path.join(save_dir, f'attention_analysis_{tag}.png')
    plt.savefig(attn_path, dpi=150); plt.close()
    print(f"\nAttention maps → {attn_path}")
