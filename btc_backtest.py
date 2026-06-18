"""
btc_backtest.py — Event-driven backtest of the Temporal Transformer signals.

Pipeline:
  1. Rebuild the full feature matrix via btc_predict.py's pipeline (no look-ahead).
  2. Run the trained checkpoint over EVERY candle -> P(Long).
  3. Apply the regime-adaptive thresholds -> LONG / SHORT / NEUTRAL.
  4. For each signal, simulate a trade with independent per-direction barriers
     (LONG_TP/LONG_SL, SHORT_TP/SHORT_SL) and a 4-candle timeout, with the
     payoff taken in the trade's natural direction.
  5. Charge fees, then report:
        - per-signal expectancy (overall, by class, by regime, by split)
        - a non-overlapping compounded equity curve vs buy-and-hold
        - Sharpe, profit factor, win rate, max drawdown.

Honesty notes:
  * Headline metrics are reported on the TEST split (out-of-sample). Train/val
    are shown too but are in-sample and will look optimistically good.
  * Ambiguous candles (TP and SL both touched) are resolved to the STOP
    (conservative).
  * Costs are configurable below; defaults assume a liquid futures taker.

Run:  py -3.10 btc_backtest.py            # test split (default)
      py -3.10 btc_backtest.py --split full
"""
import os, sys, argparse, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings('ignore')
import btc_predict as P            # reuse data pipeline + (fixed) model classes

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAVE_DIR    = r"D:\Document\LLLLLLLLLLLLL"
CKPT_PATH   = P.CKPT_PATH
# Execution barriers — independent take-profit / stop-loss per direction.
# Defaults reproduce the original setup (long 1.5/3, short 3/1.5).
LONG_TP   = 0.03    # long  take-profit (price rises this much)
LONG_SL   = 0.015     # long  stop-loss   (price falls this much)
SHORT_TP  = 0.03     # short take-profit (price FALLS this much)
SHORT_SL  = 0.015    # short stop-loss   (price RISES this much)
# Barriers used ONLY to tag the train/val/test split — keep matched to the
# triple-barrier labels the model was trained on, regardless of execution TP/SL.
LABEL_TP  = 0.015
LABEL_SL  = 0.03
TB_TIMEOUT  = 4
FEE_PER_SIDE = 0.0005     # 5 bps taker; round-trip = 0.10%
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.85
BARS_PER_YEAR = 365 * 24 / 4      # 4h candles


# ─────────────────────────────────────────────
# TRIPLE-BARRIER TRADE SIMULATION
# ─────────────────────────────────────────────
def simulate_trade(close, high, low, i, direction, n):
    """Return (gross_return, exit_idx) for a trade opened at candle i."""
    entry = close[i]
    last  = min(i + TB_TIMEOUT, n - 1)
    if direction == 'LONG':
        up = entry * (1 + LONG_TP)    # take-profit level
        dn = entry * (1 - LONG_SL)    # stop-loss level
        for j in range(i + 1, min(i + 1 + TB_TIMEOUT, n)):
            hit_tp = high[j] >= up
            hit_sl = low[j]  <= dn
            if hit_tp and hit_sl: return -LONG_SL, j         # ambiguous -> stop
            if hit_tp:            return  LONG_TP, j
            if hit_sl:            return -LONG_SL, j
        return close[last] / entry - 1.0, last               # timeout
    else:  # SHORT (profit when price falls to dn, stop when it rises to up)
        dn = entry * (1 - SHORT_TP)   # take-profit level (price drops)
        up = entry * (1 + SHORT_SL)   # stop-loss level   (price rises)
        for j in range(i + 1, min(i + 1 + TB_TIMEOUT, n)):
            hit_tp = low[j]  <= dn
            hit_sl = high[j] >= up
            if hit_tp and hit_sl: return -SHORT_SL, j        # ambiguous -> stop
            if hit_tp:            return  SHORT_TP, j
            if hit_sl:            return -SHORT_SL, j
        return -(close[last] / entry - 1.0), last            # timeout


# ─────────────────────────────────────────────
# BUILD FEATURES + SIGNALS
# ─────────────────────────────────────────────
def build():
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    cfg, scaler, feat_cols = ckpt['model_cfg'], ckpt['scaler'], ckpt['feat_cols']
    seq_len = ckpt['seq_len']
    vol_p33, vol_p67 = ckpt.get('vol_p33'), ckpt.get('vol_p67')

    model = P.TemporalTransformer(cfg['n_feat'], cfg)
    model.load_state_dict(ckpt['model_state']); model.eval()

    d4  = P.add_indicators(P.load_csv(os.path.join(P.DATA_DIR, P.HIST_FILES['4h'])),  '4h_')
    d1h = P.add_indicators(P.load_csv(os.path.join(P.DATA_DIR, P.HIST_FILES['1h'])),  '1h_')
    d1d = P.add_indicators(P.load_csv(os.path.join(P.DATA_DIR, P.HIST_FILES['1d'])),  '1d_')
    d15 = P.add_indicators(P.load_csv(os.path.join(P.DATA_DIR, P.HIST_FILES['15m'])), '15m_')
    base = d4.copy()
    for other, px in [(d1h,'1h_'), (d1d,'1d_'), (d15,'15m_')]:
        base = pd.merge_asof(base.sort_values('timestamp'),
                             other[P.ind_cols(other, px)].sort_values('timestamp'),
                             on='timestamp', direction='backward')
    base = base.sort_values('timestamp').reset_index(drop=True)
    for a, b, name in [('4h_rsi14','1d_rsi14','div_rsi_4h1d'),
                       ('4h_rsi14','1h_rsi14','div_rsi_4h1h'),
                       ('4h_macdh','1d_macdh','div_macdh_4h1d')]:
        base[name] = base[a] - base[b]
    base['div_vol_4h1d'] = base['4h_vr'] / (base['1d_vr'] + 1e-9)
    base = base.dropna(subset=feat_cols).reset_index(drop=True)

    X = np.clip(scaler.transform(base[feat_cols].values.astype(np.float32)), -6, 6).astype(np.float32)
    ts    = base['timestamp'].values
    close = base['Close'].values; high = base['High'].values; low = base['Low'].values
    n = len(base)

    # run model over every candle that has a full look-back window
    probs = np.full(n, np.nan)
    seqs, idxs = [], []
    for i in range(seq_len - 1, n):
        seqs.append(X[i - seq_len + 1:i + 1]); idxs.append(i)
        if len(seqs) == 512:
            with torch.no_grad():
                p = torch.softmax(model(torch.from_numpy(np.stack(seqs))), 1)[:, 1].numpy()
            probs[idxs] = p; seqs, idxs = [], []
    if seqs:
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(np.stack(seqs))), 1)[:, 1].numpy()
        probs[idxs] = p

    # regime-adaptive thresholds (same logic as btc_predict.py)
    vol_idx = feat_cols.index(P.VOL_FEATURE)
    vol = X[:, vol_idx]
    if vol_p33 is None:
        vol_p33, vol_p67 = np.percentile(vol, [33, 67])
    long_thr  = np.where(vol > vol_p67, P.THRESH_HIGH_VOL[0],
                np.where(vol < vol_p33, P.THRESH_LOW_VOL[0], P.LONG_THRESH))
    short_thr = np.where(vol > vol_p67, P.THRESH_HIGH_VOL[1],
                np.where(vol < vol_p33, P.THRESH_LOW_VOL[1], P.SHORT_THRESH))
    regime = np.where(vol > vol_p67, 'high', np.where(vol < vol_p33, 'low', 'mid'))
    sig = np.where(probs >= long_thr, 'LONG', np.where(probs <= short_thr, 'SHORT', 'NEUTRAL'))
    sig[np.isnan(probs)] = 'NEUTRAL'

    # chronological split boundaries (match training: split labeled positions 70/85)
    # label presence == has a resolved barrier within horizon (for split tagging only)
    has_label = np.zeros(n, bool)
    for i in range(n - 1):
        e = close[i]; up = e*(1+LABEL_TP); dn = e*(1-LABEL_SL)
        for j in range(i+1, min(i+1+TB_TIMEOUT, n)):
            if high[j] >= up or low[j] <= dn:
                has_label[i] = True; break
    lab_idx = np.where(has_label)[0]
    i_tr = lab_idx[int(len(lab_idx)*TRAIN_FRAC)]
    i_va = lab_idx[int(len(lab_idx)*VAL_FRAC)]
    split = np.where(np.arange(n) < i_tr, 'train',
            np.where(np.arange(n) < i_va, 'val', 'test'))

    return dict(ts=ts, close=close, high=high, low=low, n=n,
                probs=probs, sig=sig, regime=regime, split=split)


# ─────────────────────────────────────────────
# RUN TRADES
# ─────────────────────────────────────────────
def run_trades(d, split_filter):
    ts, close, high, low, n = d['ts'], d['close'], d['high'], d['low'], d['n']
    sig, regime, split = d['sig'], d['regime'], d['split']

    rows = []
    for i in range(n):
        if sig[i] in ('LONG', 'SHORT') and (split_filter == 'full' or split[i] == split_filter):
            gross, j = simulate_trade(close, high, low, i, sig[i], n)
            net = gross - 2 * FEE_PER_SIDE
            rows.append(dict(entry_ts=ts[i], exit_ts=ts[j], direction=sig[i],
                             regime=regime[i], prob_long=d['probs'][i],
                             gross=gross, net=net, exit_idx=j, entry_idx=i))
    return pd.DataFrame(rows)


def equity_curve(trades, d):
    """Non-overlapping, single-unit-capital, compounded."""
    trades = trades.sort_values('entry_idx').reset_index(drop=True)
    equity, eq_ts, eq_val = 1.0, [], []
    free_at = -1
    used = []
    for _, t in trades.iterrows():
        if t.entry_idx <= free_at:
            continue                      # still in a trade
        equity *= (1 + t.net)
        free_at = t.exit_idx
        eq_ts.append(t.exit_ts); eq_val.append(equity)
        used.append(t)
    return pd.DataFrame({'timestamp': eq_ts, 'equity': eq_val}), pd.DataFrame(used)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def stats(name, tr):
    if len(tr) == 0:
        print(f"  {name:18s}  no trades"); return
    wr   = (tr.net > 0).mean()
    exp  = tr.net.mean()
    gp   = tr.net[tr.net > 0].sum(); gl = -tr.net[tr.net < 0].sum()
    pf   = gp / gl if gl > 0 else float('inf')
    sharpe = tr.net.mean() / (tr.net.std() + 1e-9) * np.sqrt(len(tr) / max((tr.entry_idx.iloc[-1]-tr.entry_idx.iloc[0])/BARS_PER_YEAR, 1e-9)) if len(tr) > 2 else 0
    print(f"  {name:18s}  n={len(tr):4d}  win%={wr*100:5.1f}  "
          f"exp/trade={exp*100:+6.3f}%  PF={pf:4.2f}  totedge={tr.net.sum()*100:+7.2f}%")


def max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak).min()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'val', 'train', 'full'])
    args = ap.parse_args()

    print("Building features + running model over all candles...")
    d = build()
    span = f"{pd.Timestamp(d['ts'][0]):%Y-%m-%d} -> {pd.Timestamp(d['ts'][-1]):%Y-%m-%d}"
    n_test = (d['split'] == 'test').sum()
    print(f"  candles: {d['n']:,}   span: {span}")
    print(f"  split sizes -> train:{(d['split']=='train').sum():,}  "
          f"val:{(d['split']=='val').sum():,}  test:{n_test:,}")
    print(f"  fees: {FEE_PER_SIDE*1e4:.0f} bps/side  ({2*FEE_PER_SIDE*100:.2f}% round-trip)")

    trades = run_trades(d, args.split)
    print(f"\n=== PER-SIGNAL EXPECTANCY  (split = {args.split}) ===")
    stats('ALL', trades)
    stats('LONG',  trades[trades.direction == 'LONG'])
    stats('SHORT', trades[trades.direction == 'SHORT'])
    for reg in ('low', 'mid', 'high'):
        stats(f'regime={reg}', trades[trades.regime == reg])

    eq, used = equity_curve(trades, d)
    if len(eq) > 1:
        mdd = max_drawdown(eq['equity'].values)
        years = (eq['timestamp'].iloc[-1] - eq['timestamp'].iloc[0]) / np.timedelta64(365, 'D')
        cagr = eq['equity'].iloc[-1] ** (1/max(years, 1e-9)) - 1
        # buy & hold over same window
        m = (d['split'] == args.split) if args.split != 'full' else np.ones(d['n'], bool)
        c = d['close'][m]
        bh = c[-1] / c[0] - 1
        print(f"\n=== STRATEGY EQUITY  (non-overlapping, compounded) ===")
        print(f"  trades taken : {len(used)}")
        print(f"  total return : {(eq['equity'].iloc[-1]-1)*100:+.2f}%")
        print(f"  CAGR         : {cagr*100:+.2f}%  over {years:.2f} yr")
        print(f"  max drawdown : {mdd*100:.2f}%")
        print(f"  buy & hold   : {bh*100:+.2f}%  (same window)")

        plt.figure(figsize=(11, 5))
        plt.plot(eq['timestamp'], eq['equity'], label='Strategy', color='steelblue')
        bh_curve = c / c[0]
        plt.plot(pd.to_datetime(d['ts'][m]), bh_curve, label='Buy & Hold', color='gray', alpha=0.7)
        plt.axhline(1.0, color='k', lw=0.6, ls=':')
        plt.title(f'BTC Transformer Backtest — {args.split} split  ({2*FEE_PER_SIDE*100:.2f}% round-trip)')
        plt.ylabel('Equity (×)'); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(SAVE_DIR, f'backtest_{args.split}.png')
        plt.savefig(out_png, dpi=150); plt.close()
        print(f"  equity plot  -> {out_png}")

    trades.to_csv(os.path.join(SAVE_DIR, f'backtest_trades_{args.split}.csv'), index=False)
    print(f"\nSaved trades -> backtest_trades_{args.split}.csv")


if __name__ == '__main__':
    main()
