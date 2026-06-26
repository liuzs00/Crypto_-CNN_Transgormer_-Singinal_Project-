"""
btc_predict_cross_nf.py — Live BTC signal for the PRODUCTION model
`btc_eth_sol_pooled_model_emb_cross_nf.pth` (BTC+ETH+SOL pooled transformer + asset
embedding + cross-asset / absorption / anchored-VWAP / path-entropy features + temporal
conv-stem tokenizer).

On every run it **auto-fills the data gap**: it reads the last timestamp across all model
CSVs (BTC/ETH/SOL × 15m/1h/4h/1d), then calls `crypto_data.py --mode update` with a window
sized to cover the gap — which de-duplicates repeated rows and validates continuity. Then it
REUSES the exact feature pipeline + model + thresholds from `btc_backtest_cross_nf.py`
(`build()`), so there is one inference path with no train/serve parity drift.

Default  : signals for the last 24 hours of available data
Custom   : any date range or lookback window via CLI flags

Usage:
    py -3.10 btc_predict_cross_nf.py                          # auto-update, last 24 h
    py -3.10 btc_predict_cross_nf.py --days 7                 # last 7 days
    py -3.10 btc_predict_cross_nf.py --from 2026-06-01        # from date to latest
    py -3.10 btc_predict_cross_nf.py --from 2026-06-01 --to 2026-06-10
    py -3.10 btc_predict_cross_nf.py --no-update              # skip the data refresh

Output: btc_predict_cross_nf_signals.csv  —  timestamp, close, prob_long, vol_regime, signal
"""
import os, sys, argparse, subprocess
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd
import btc_backtest_cross_nf as BT     # reuse build(): feature pipeline + model + thresholds
import btc_predict as P                # parse_dt + regime-threshold constants

OUT_CSV = os.path.join(BT.SAVE_DIR, 'btc_predict_cross_nf_signals.csv')
THR = {'low': P.THRESH_LOW_VOL, 'mid': (P.LONG_THRESH, P.SHORT_THRESH), 'high': P.THRESH_HIGH_VOL}


# ── data-gap auto-fill ────────────────────────────────────────────
def _last_timestamp(path):
    """Read just the final line of a CSV (seek from end) → its Open-time as UTC Timestamp."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode('utf-8', 'ignore')
        line = tail.strip().splitlines()[-1]
        return pd.to_datetime(line.split(',')[0].strip(), utc=True, errors='coerce')
    except (OSError, IndexError):
        return pd.NaT


def fill_data_gap(buffer_days=2):
    """Size an update window from the oldest 'last candle' across all model CSVs, then let
    crypto_data.py fetch + merge it (handles de-dup of repeated rows + continuity check)."""
    files = [os.path.join(BT.DATA_DIR, tf_files[tf])
             for tf_files in BT.ASSETS.values() for tf in tf_files]
    lasts = [t for t in map(_last_timestamp, files) if pd.notna(t)]
    if not lasts:
        print("  no data CSVs found — skipping auto-update."); return
    oldest = min(lasts)
    gap_days = (pd.Timestamp.now(tz='UTC') - oldest).total_seconds() / 86400.0
    days = max(2, int(np.ceil(gap_days)) + buffer_days)
    print(f"  oldest 'last candle' across CSVs: {oldest:%Y-%m-%d %H:%M} UTC  "
          f"(gap ≈ {gap_days:.1f}d) → fetching last {days}d to fill it\n")
    try:
        subprocess.run([sys.executable, os.path.join(BT.SAVE_DIR, 'crypto_data.py'),
                        '--mode', 'update', '--days', str(days)], check=True)
    except Exception as e:
        print(f"  ! auto-update failed ({e}); predicting on existing data.")


# ── CLI ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Live BTC signal — production emb_cross_nf model")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--days', type=float, default=1.0,
                     help='look back N days from the latest candle (default 1.0 = last 24 h)')
    grp.add_argument('--from', dest='date_from', metavar='DATETIME',
                     help='start of window  e.g. 2026-06-01  or  "2026-06-01 08:00"')
    p.add_argument('--to', dest='date_to', metavar='DATETIME', default=None,
                   help='end of window (default: latest available candle)')
    p.add_argument('--no-update', action='store_true', help='skip the automatic data refresh')
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_update:
        print("Auto-filling data gap (crypto_data.py update — de-dup + continuity) ...")
        fill_data_gap()
        print()

    print(f"Model: {os.path.basename(BT.CKPT_PATH)}")
    print("       (emb + cross-asset + VWAP + Absorption + path-entropy + conv-stem, BTC asset_id=0)")
    d = BT.build()                          # full pipeline + inference over all candles

    n, probs, sig, regime, close = d['n'], d['probs'], d['sig'], d['regime'], d['close']
    ts_all = pd.to_datetime(d['ts'])        # tz-naive UTC

    last = n - 1
    while last > 0 and np.isnan(probs[last]):
        last -= 1
    latest = ts_all[last]

    # resolve the display window
    if args.date_from:
        t_from = P.parse_dt(args.date_from).tz_localize(None)
        t_to   = P.parse_dt(args.date_to).tz_localize(None) if args.date_to else latest
        label  = f"{t_from:%Y-%m-%d %H:%M} → {t_to:%Y-%m-%d %H:%M} UTC"
    else:
        t_to, t_from = latest, latest - pd.Timedelta(days=args.days)
        label = f"last {args.days:g} day(s)"

    rows = [i for i in range(n)
            if t_from <= ts_all[i] <= t_to and not np.isnan(probs[i])]
    if not rows:
        print(f"\nNo candles with valid signals in the requested window ({label}).")
        return
    tip = rows[-1]                          # most recent candle in the window

    print(f"\n  candles: {n:,}   latest scored: {latest:%Y-%m-%d %H:%M} UTC   close: {close[last]:,.2f}")
    print("\n" + "=" * 70)
    print(f"  LATEST SIGNAL  →  {sig[tip]}     P(long)={probs[tip]:.4f}   regime={regime[tip]}")
    print("=" * 70)

    print(f"\n  Signals — {label}  ({len(rows)} candles):")
    print(f"  {'Timestamp (UTC)':<18} {'Close':>11} {'P_long':>8} {'Regime':>7} {'L/S thr':>9}  Signal")
    print(f"  {'-'*70}")
    for i in rows:
        lt, st = THR[regime[i]]
        mark = '  *' if sig[i] != 'NEUTRAL' else ''
        print(f"  {ts_all[i]:%Y-%m-%d %H:%M} {close[i]:>11,.2f} {probs[i]:>8.4f} "
              f"{regime[i]:>7} {lt:>4.2f}/{st:<4.2f} {sig[i]}{mark}")

    pd.DataFrame({
        'timestamp':  [ts_all[i] for i in rows],
        'close':      [close[i] for i in rows],
        'prob_long':  [probs[i] for i in rows],
        'vol_regime': [regime[i] for i in rows],
        'signal':     [sig[i] for i in rows],
    }).to_csv(OUT_CSV, index=False)
    n_act = sum(sig[i] != 'NEUTRAL' for i in rows)
    print(f"\n  Active in window: {n_act}/{len(rows)}   Saved → {os.path.basename(OUT_CSV)}")


if __name__ == '__main__':
    main()
