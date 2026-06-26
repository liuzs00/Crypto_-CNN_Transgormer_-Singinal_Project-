"""
btc_backtest_cross_nf.py — Backtest the production model
btc_eth_sol_pooled_model_emb_cross_nf.pth (BTC+ETH+SOL pooled conv-stem transformer +
asset embedding + cross-asset + Absorption + Anchored-VWAP + path-entropy features).

Feature pipeline and model are IMPORTED from the shared modules (crypto_features /
crypto_model) — the exact same code the trainer uses, so there is no parity drift to
chase. Reuses btc_backtest.py for the triple-barrier trade simulation + metrics.
btc_predict_cross_nf.py calls build() from here for live signals.

Run:  py -3.10 btc_backtest_cross_nf.py            # BTC test split (default)
      CKPT=<path> py -3.10 btc_backtest_cross_nf.py --split full
"""
import os, sys, argparse, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch
warnings.filterwarnings('ignore')
import btc_backtest as B            # simulate_trade, run_trades, equity_curve, stats, max_drawdown, costs
import btc_predict as P             # THRESH_*, VOL_FEATURE
import crypto_features as FE        # SHARED feature pipeline (load/add_indicators/cross/absorption)
from crypto_model import TemporalTransformer   # SHARED model definition

SAVE_DIR  = r"D:\Document\LLLLLLLLLLLLL"
DATA_DIR  = FE.DATA_DIR             # alias (btc_predict_cross_nf.py reads BT.DATA_DIR / BT.ASSETS)
ASSETS    = FE.ASSETS
CKPT_PATH = os.environ.get('CKPT', os.path.join(SAVE_DIR, 'btc_eth_sol_pooled_model_emb_cross_nf.pth'))
TRAIN_FRAC, VAL_FRAC = 0.70, 0.85


def build():
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    cfg, scaler, feat_cols = ckpt['model_cfg'], ckpt['scaler'], ckpt['feat_cols']
    seq_len = ckpt['seq_len']; vp33, vp67 = ckpt.get('vol_p33'), ckpt.get('vol_p67')
    model = TemporalTransformer.from_cfg(cfg); model.load_state_dict(ckpt['model_state']); model.eval()
    print(f"  model n_feat={cfg['n_feat']}  use_emb={cfg.get('use_emb')}  use_convstem={cfg.get('use_convstem', False)}")

    print("  building BTC+ETH+SOL features (this includes the absorption rolling-PCA)...")
    bases = {a: FE.build_base(f) for a, f in FE.ASSETS.items()}
    bases = FE.add_cross_asset(bases); bases = FE.add_absorption(bases)
    btc = bases['btc'].dropna(subset=feat_cols).reset_index(drop=True)
    X = np.clip(scaler.transform(btc[feat_cols].values.astype(np.float32)), -6, 6).astype(np.float32)
    ts = btc['timestamp'].values; close = btc['Close'].values; high = btc['High'].values; low = btc['Low'].values; n = len(btc)

    probs = np.full(n, np.nan); seqs, idxs = [], []
    for i in range(seq_len-1, n):
        seqs.append(X[i-seq_len+1:i+1]); idxs.append(i)
        if len(seqs) == 512:
            xb = torch.from_numpy(np.stack(seqs)); aid = torch.zeros(len(seqs), dtype=torch.int64)
            with torch.no_grad(): probs[idxs] = torch.softmax(model(xb, aid), 1)[:, 1].numpy()
            seqs, idxs = [], []
    if seqs:
        xb = torch.from_numpy(np.stack(seqs)); aid = torch.zeros(len(seqs), dtype=torch.int64)
        with torch.no_grad(): probs[idxs] = torch.softmax(model(xb, aid), 1)[:, 1].numpy()

    vol_idx = feat_cols.index(P.VOL_FEATURE); vol = X[:, vol_idx]
    if vp33 is None: vp33, vp67 = np.percentile(vol, [33, 67])
    long_thr  = np.where(vol > vp67, P.THRESH_HIGH_VOL[0], np.where(vol < vp33, P.THRESH_LOW_VOL[0], P.LONG_THRESH))
    short_thr = np.where(vol > vp67, P.THRESH_HIGH_VOL[1], np.where(vol < vp33, P.THRESH_LOW_VOL[1], P.SHORT_THRESH))
    regime = np.where(vol > vp67, 'high', np.where(vol < vp33, 'low', 'mid'))
    sig = np.where(probs >= long_thr, 'LONG', np.where(probs <= short_thr, 'SHORT', 'NEUTRAL')); sig[np.isnan(probs)] = 'NEUTRAL'

    lab = FE.triple_barrier(close, high, low, FE.LONG_TP, FE.LONG_SL, FE.SHORT_TP, FE.SHORT_SL, FE.TB_TIMEOUT); valid = ~np.isnan(lab)
    lab_end = np.array([i for i in range(seq_len, n+1) if valid[i-1]])
    t_tr = lab_end[int(len(lab_end)*TRAIN_FRAC)]-1; t_va = lab_end[int(len(lab_end)*VAL_FRAC)]-1
    split = np.where(np.arange(n) < t_tr, 'train', np.where(np.arange(n) < t_va, 'val', 'test'))
    return dict(ts=ts, close=close, high=high, low=low, n=n, probs=probs, sig=sig, regime=regime, split=split)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test', choices=['test', 'val', 'train', 'full']); args = ap.parse_args()
    print(f"Model: {os.path.basename(CKPT_PATH)}  (emb + cross-asset + VWAP + Absorption + entropy + conv-stem, BTC asset_id=0)")
    d = build()
    span = f"{pd.Timestamp(d['ts'][0]):%Y-%m-%d} -> {pd.Timestamp(d['ts'][-1]):%Y-%m-%d}"
    print(f"  candles:{d['n']:,}  span:{span}  fees:{B.FEE_PER_SIDE*1e4:.0f}bps/side")
    print(f"  split sizes -> train:{(d['split']=='train').sum():,}  val:{(d['split']=='val').sum():,}  test:{(d['split']=='test').sum():,}")
    trades = B.run_trades(d, args.split)
    print(f"\n=== PER-SIGNAL EXPECTANCY  (split={args.split}) ===")
    B.stats('ALL', trades); B.stats('LONG', trades[trades.direction == 'LONG']); B.stats('SHORT', trades[trades.direction == 'SHORT'])
    for reg in ('low', 'mid', 'high'): B.stats(f'regime={reg}', trades[trades.regime == reg])
    eq, used = B.equity_curve(trades, d)
    if len(eq) > 1:
        mdd = B.max_drawdown(eq['equity'].values); years = (eq['timestamp'].iloc[-1]-eq['timestamp'].iloc[0])/np.timedelta64(365, 'D')
        cagr = eq['equity'].iloc[-1]**(1/max(years, 1e-9))-1
        m = (d['split'] == args.split) if args.split != 'full' else np.ones(d['n'], bool); c = d['close'][m]; bh = c[-1]/c[0]-1
        print(f"\n=== STRATEGY EQUITY (non-overlapping, compounded) ===")
        print(f"  trades taken:{len(used)}  total return:{(eq['equity'].iloc[-1]-1)*100:+.2f}%  CAGR:{cagr*100:+.2f}%  maxDD:{mdd*100:.2f}%")
        print(f"  buy & hold  :{bh*100:+.2f}% (same window)")
        plt.figure(figsize=(11, 5)); plt.plot(eq['timestamp'], eq['equity'], label='Strategy (emb_cross_nf)', color='darkgreen')
        plt.plot(pd.to_datetime(d['ts'][m]), c/c[0], label='Buy & Hold', color='gray', alpha=0.7); plt.axhline(1.0, color='k', lw=0.6, ls=':')
        plt.title(f'Newest model backtest — {args.split}  ({2*B.FEE_PER_SIDE*100:.2f}% round-trip)'); plt.ylabel('Equity (×)'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        out = os.path.join(SAVE_DIR, f'backtest_cross_nf_{args.split}.png'); plt.savefig(out, dpi=150); plt.close(); print(f"  equity plot -> {out}")
    trades.to_csv(os.path.join(SAVE_DIR, f'backtest_cross_nf_trades_{args.split}.csv'), index=False)
    print(f"\nSaved trades -> backtest_cross_nf_trades_{args.split}.csv")


if __name__ == '__main__':
    main()
