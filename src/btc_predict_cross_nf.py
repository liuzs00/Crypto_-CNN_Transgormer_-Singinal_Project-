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

Each active signal also carries an **M2 meta-label** (btc_meta_label.py): a confidence score
and a continuous **recommended size** (0 = skip, 1 = full position) that scales exposure by
how likely M2 thinks the trade is to be profitable. If the M2 model file is absent the tool
still runs (size column shows "-").

Outputs:
  btc_predict_cross_nf_signals.csv  — transient VIEW of the requested window (overwritten each run):
        timestamp, close, prob_long, vol_regime, signal, m2_confidence, recommended_size
  btc_signal_state_log.csv          — persistent APPEND-ONLY history: each run adds only the new
        bar(s) (never rewrites/loses stored signals) → a live track-record + source for
        conviction-dynamics features.
"""
import os, sys, argparse, subprocess
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd
import btc_backtest_cross_nf as BT     # reuse build(): feature pipeline + model + thresholds
import btc_predict as P                # parse_dt + regime-threshold constants
import btc_meta_label as ML            # M2 meta-label: confidence + recommended size

from paths import OUTPUTS_DIR, LOGS_DIR, SRC_DIR
OUT_CSV = os.path.join(OUTPUTS_DIR, 'btc_predict_cross_nf_signals.csv')       # transient window VIEW (overwritten each run)
SIGNAL_LOG = os.path.join(LOGS_DIR, 'btc_signal_state_log.csv')               # persistent append-only signal-state HISTORY
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
        subprocess.run([sys.executable, os.path.join(SRC_DIR, 'crypto_data.py'),
                        '--mode', 'update', '--days', str(days)], check=True)
    except Exception as e:
        print(f"  ! auto-update failed ({e}); predicting on existing data.")
    try:                                          # daily NQ/ES macro source (yfinance) — degrades gracefully
        subprocess.run([sys.executable, os.path.join(SRC_DIR, 'macro_data.py'), '--daily-only'], check=True)
    except Exception as e:
        print(f"  ! macro (NQ/ES) update failed ({e}); macro features fall back to last-known / neutral.")


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


def update_signal_log(out_df):
    """Append ONLY genuinely-new bars to the persistent signal-state log (SIGNAL_LOG).

    Unlike OUT_CSV (a transient last-N-hours view that is fully overwritten each run), this is an
    APPEND-ONLY history: each auto-run adds only the bar(s) whose timestamp is beyond the last one
    already logged (steady state = one new 4h bar per run; also backfills a short gap after downtime)
    and NEVER rewrites or drops previously stored signals. It is the durable {direction, M1conf
    (prob_long), vol_regime, M2conf, sizing} record — a live track-record and the source series for
    any future conviction-dynamics features."""
    out_df = out_df.copy()
    out_df['timestamp'] = pd.to_datetime(out_df['timestamp'], utc=True, errors='coerce')
    out_df = out_df.dropna(subset=['timestamp']).sort_values('timestamp')
    last_ts = None
    if os.path.exists(SIGNAL_LOG):
        try:
            prev = pd.read_csv(SIGNAL_LOG)
            last_ts = pd.to_datetime(prev['timestamp'], utc=True, errors='coerce').dropna().max()
            if pd.isna(last_ts): last_ts = None
        except Exception:
            last_ts = None
    new = out_df if last_ts is None else out_df[out_df['timestamp'] > last_ts]
    if len(new) == 0:
        print(f"  Signal log: no new bar since {last_ts:%Y-%m-%d %H:%M} UTC → "
              f"{os.path.basename(SIGNAL_LOG)} unchanged (history preserved)")
        return
    new.to_csv(SIGNAL_LOG, mode='a', header=not os.path.exists(SIGNAL_LOG), index=False)
    print(f"  Signal log: +{len(new)} new bar(s) (through {new['timestamp'].iloc[-1]:%Y-%m-%d %H:%M} UTC) "
          f"→ {os.path.basename(SIGNAL_LOG)}  (append-only, {'' if last_ts is None else 'prior history kept'})")


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

    # ── M2 meta-label: confidence + recommended size for active signals (frozen M1 + M2) ──
    m2conf = {i: np.nan for i in rows}; m2size = {i: 0.0 for i in rows}; m2_on = False
    act = [i for i in rows if sig[i] != 'NEUTRAL']
    try:
        ml = ML.load_m2()
        if act:
            is_long = np.array([sig[i] == 'LONG' for i in act])
            _cd = any(c.startswith('cd_') for c in ml['feat_order'])   # match the saved M2's feature set
            Xm = ML.m2_features(d, act, is_long, np.array([probs[i] for i in act]), convdyn=_cd)[ml['feat_order']]
            pc = ml['m2'].predict_proba(Xm)[:, 1]
            for k, i in enumerate(act):
                m2conf[i] = float(pc[k]); m2size[i] = ML.recommended_size(pc[k], ml['skip_thr'], ml['full_thr'])
        m2_on = True
    except Exception as e:
        print(f"  (M2 meta-label unavailable — run btc_meta_label.py to train it: {e})")

    def _m2cell(i):
        if sig[i] != 'NEUTRAL' and m2_on and not np.isnan(m2conf[i]):
            return f"{m2conf[i]:.3f}", f"{m2size[i]:.2f}"
        return "  -  ", "  - "

    tip_x = ""
    if sig[tip] != 'NEUTRAL' and m2_on and not np.isnan(m2conf[tip]):
        tip_x = f"   M2conf={m2conf[tip]:.3f}  size={m2size[tip]:.2f}×"
    print(f"\n  candles: {n:,}   latest scored: {latest:%Y-%m-%d %H:%M} UTC   close: {close[last]:,.2f}")
    print("\n" + "=" * 78)
    print(f"  LATEST SIGNAL  →  {sig[tip]}     P(long)={probs[tip]:.4f}   regime={regime[tip]}{tip_x}")
    print("=" * 78)

    print(f"\n  Signals — {label}  ({len(rows)} candles):")
    print(f"  {'Timestamp (UTC)':<18} {'Close':>11} {'P_long':>8} {'Regime':>7} {'L/S thr':>9} {'Signal':<8} {'M2conf':>7} {'Size':>5}")
    print(f"  {'-'*78}")
    for i in rows:
        lt, st = THR[regime[i]]; mc, sz = _m2cell(i)
        print(f"  {ts_all[i]:%Y-%m-%d %H:%M} {close[i]:>11,.2f} {probs[i]:>8.4f} "
              f"{regime[i]:>7} {lt:>4.2f}/{st:<4.2f} {sig[i]:<8} {mc:>7} {sz:>5}")

    out_df = pd.DataFrame({
        'timestamp':        [ts_all[i] for i in rows],
        'close':            [close[i] for i in rows],
        'prob_long':        [probs[i] for i in rows],
        'vol_regime':       [regime[i] for i in rows],
        'signal':           [sig[i] for i in rows],
        'm2_confidence':    [m2conf[i] for i in rows],
        'recommended_size': [m2size[i] for i in rows],
    })
    out_df.to_csv(OUT_CSV, index=False)                 # transient window view (overwritten)
    update_signal_log(out_df)                            # persistent append-only signal-state history
    n_act = sum(sig[i] != 'NEUTRAL' for i in rows)
    print(f"\n  Active in window: {n_act}/{len(rows)}   (Size = fraction of full position, M2-scaled)")
    print(f"  Saved → {os.path.basename(OUT_CSV)}")


if __name__ == '__main__':
    main()
