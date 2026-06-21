"""
btc_predict_cross_nf.py — Live BTC signal for the PRODUCTION model
`btc_eth_sol_pooled_model_emb_cross_nf.pth` (BTC+ETH+SOL pooled transformer + asset
embedding + cross-asset / absorption / anchored-VWAP / path-entropy features + temporal
conv-stem tokenizer).

It REUSES the exact feature pipeline + model + regime-adaptive thresholds from
`btc_backtest_cross_nf.py` (`build()`), so there is no second copy of the inference logic to
drift out of train/serve parity. It then prints the most recent N BTC signals, with the
latest one highlighted, and saves them to CSV.

Because of the cross-asset + absorption features, BTC inference needs **live ETH and SOL
data** too — use `--update` (or run `crypto_data.py` first) to refresh all three.

Run:  py -3.10 btc_predict_cross_nf.py                # last 15 candles, current CSVs
      py -3.10 btc_predict_cross_nf.py --n 40         # last 40 candles
      py -3.10 btc_predict_cross_nf.py --update       # refresh BTC/ETH/SOL data, then predict
      py -3.10 btc_predict_cross_nf.py --split full    # score the whole history (default: recent)
"""
import os, sys, argparse, subprocess
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd
import btc_backtest_cross_nf as BT     # reuse build(): feature pipeline + model + thresholds
import btc_predict as P                # regime-threshold constants

OUT_CSV = os.path.join(BT.SAVE_DIR, 'btc_predict_cross_nf_signals.csv')
THR = {'low': P.THRESH_LOW_VOL, 'mid': (P.LONG_THRESH, P.SHORT_THRESH), 'high': P.THRESH_HIGH_VOL}


def main():
    ap = argparse.ArgumentParser(description="Live BTC signal — production emb_cross_nf model")
    ap.add_argument('--n', type=int, default=15, help="how many most-recent candles to display")
    ap.add_argument('--update', action='store_true', help="refresh BTC/ETH/SOL data first (crypto_data.py)")
    args = ap.parse_args()

    if args.update:
        print("Refreshing data (crypto_data.py --mode update) ...\n")
        subprocess.run([sys.executable, os.path.join(BT.SAVE_DIR, 'crypto_data.py'), '--mode', 'update'], check=True)

    print(f"Model: {os.path.basename(BT.CKPT_PATH)}")
    print("       (emb + cross-asset + VWAP + Absorption + path-entropy + conv-stem, BTC asset_id=0)")
    d = BT.build()                          # full pipeline + inference over all candles

    n, probs, sig, regime = d['n'], d['probs'], d['sig'], d['regime']
    ts, close = d['ts'], d['close']

    # latest candle with a valid (non-NaN) probability = the live signal
    last = n - 1
    while last > 0 and np.isnan(probs[last]):
        last -= 1
    k = min(args.n, last + 1)

    print(f"\n  candles: {n:,}   latest scored: {pd.Timestamp(ts[last]):%Y-%m-%d %H:%M} UTC"
          f"   close: {close[last]:,.2f}")
    print("\n" + "=" * 70)
    print(f"  LATEST SIGNAL  →  {sig[last]}     P(long)={probs[last]:.4f}   regime={regime[last]}")
    print("=" * 70)

    print(f"\n  Last {k} candles:")
    print(f"  {'Timestamp (UTC)':<18} {'Close':>11} {'P_long':>8} {'Regime':>7} {'L/S thr':>9}  Signal")
    print(f"  {'-'*70}")
    for i in range(last - k + 1, last + 1):
        lt, st = THR[regime[i]]
        p = probs[i]
        pstr = f"{p:.4f}" if not np.isnan(p) else "  --  "
        mark = '  *' if sig[i] != 'NEUTRAL' else ''
        print(f"  {pd.Timestamp(ts[i]):%Y-%m-%d %H:%M} {close[i]:>11,.2f} {pstr:>8} "
              f"{regime[i]:>7} {lt:>4.2f}/{st:<4.2f} {sig[i]}{mark}")

    rows = range(last - k + 1, last + 1)
    pd.DataFrame({
        'timestamp':  [pd.Timestamp(ts[i]) for i in rows],
        'close':      [close[i] for i in rows],
        'prob_long':  [probs[i] for i in rows],
        'vol_regime': [regime[i] for i in rows],
        'signal':     [sig[i] for i in rows],
    }).to_csv(OUT_CSV, index=False)
    n_act = sum(sig[i] != 'NEUTRAL' for i in rows)
    print(f"\n  Active in window: {n_act}/{k}   Saved → {os.path.basename(OUT_CSV)}")


if __name__ == '__main__':
    main()
