"""
btc_predict.py — BTC trading signal prediction

Default  : signals for the last 24 hours of available data
Custom   : any date range or lookback window via CLI flags

Usage:
    python btc_predict.py                          # last 24 h (default)
    python btc_predict.py --days 7                 # last 7 days
    python btc_predict.py --from 2026-06-01        # from date to latest
    python btc_predict.py --from 2026-06-01 --to 2026-06-10
    python btc_predict.py --from "2026-06-01 08:00" --to "2026-06-03 20:00"

Output:
    btc_recent_signals.csv  —  timestamp, prob_long, prob_short, signal, vol_regime
"""

import os, warnings, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR  = r"D:\Document\LLLLLLLLLLLLL\DATA"
SAVE_DIR  = r"D:\Document\LLLLLLLLLLLLL"
CKPT_PATH = os.path.join(SAVE_DIR, 'btc_lstm_model.pth')
OUT_CSV   = os.path.join(SAVE_DIR, 'btc_recent_signals.csv')

VOL_FEATURE     = '4h_atr14'
LONG_THRESH     = 0.60
SHORT_THRESH    = 0.40
THRESH_LOW_VOL  = (0.55, 0.45)
THRESH_HIGH_VOL = (0.72, 0.28)

HIST_FILES = {
    '4h' : 'btc_4h_data_2018_to_2025.csv',
    '1h' : 'btc_1h_data_2018_to_2025.csv',
    '15m': 'btc_15m_data_2018_to_2025.csv',
    '1d' : 'btc_1d_data_2018_to_2025.csv',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="BTC signal prediction")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        '--days', type=float, default=1.0,
        help='Look back N days from the latest available candle (default: 1.0 = last 24 h)',
    )
    grp.add_argument(
        '--from', dest='date_from', metavar='DATETIME',
        help='Start of prediction window  e.g. 2026-06-01  or  "2026-06-01 08:00"',
    )
    p.add_argument(
        '--to', dest='date_to', metavar='DATETIME', default=None,
        help='End of prediction window (default: latest available candle)',
    )
    return p.parse_args()


def parse_dt(s: str) -> pd.Timestamp:
    """Accept YYYY-MM-DD or YYYY-MM-DD HH:MM, return tz-aware UTC Timestamp."""
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return pd.Timestamp(datetime.strptime(s.strip(), fmt), tz='UTC')
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{s}'. Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM'")


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════
class SqueezeExcite(nn.Module):
    def __init__(self, n_feat, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_feat, max(n_feat // r, 16)), nn.ReLU(),
            nn.Linear(max(n_feat // r, 16), n_feat), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x.mean(1)).unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, n_feat, patch_size, d_model):
        super().__init__()
        self.p    = patch_size
        self.proj = nn.Linear(n_feat * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B, T, F = x.shape
        pad = (self.p - T % self.p) % self.p
        if pad: x = F.pad(x, (0, 0, 0, pad))
        return self.norm(self.proj(x.reshape(B, -1, self.p * F)))


class RoPEEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv)
    def forward(self, L, device):
        t = torch.arange(L, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def _apply_rope(q, k, cos, sin):
    cos = cos[None, None]; sin = sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

def _alibi_slopes(n_heads):
    def slopes(n): return [2 ** (-8 * i / n) for i in range(1, n + 1)]
    if (n_heads & (n_heads - 1)) == 0:
        return torch.tensor(slopes(n_heads), dtype=torch.float32)
    p = 2 ** int(np.floor(np.log2(n_heads)))
    return torch.tensor(slopes(p) + slopes(2 * p)[0::2][:n_heads - p], dtype=torch.float32)

def _alibi_bias(n_heads, L, device):
    slopes = _alibi_slopes(n_heads).to(device)
    dist   = (torch.arange(L, device=device).unsqueeze(0) -
              torch.arange(L, device=device).unsqueeze(1)).abs().float()
    return -slopes.view(-1, 1, 1) * dist.unsqueeze(0)


class RelativeAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads; self.dh = d_model // n_heads; self.scl = self.dh ** -0.5
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RoPEEmbedding(self.dh)
    def forward(self, x):
        B, L, D = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, L, self.H, self.dh).transpose(1, 2)
        K = K.view(B, L, self.H, self.dh).transpose(1, 2)
        V = V.view(B, L, self.H, self.dh).transpose(1, 2)
        cos, sin = self.rope(L, x.device)
        Q, K = _apply_rope(Q, K, cos, sin)
        logits = (Q @ K.transpose(-2, -1)) * self.scl + _alibi_bias(self.H, L, x.device)
        return self.out((self.drop(logits.softmax(dim=-1)) @ V).transpose(1, 2).reshape(B, L, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n1   = nn.LayerNorm(d_model)
        self.attn = RelativeAttention(d_model, n_heads, dropout)
        self.n2   = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.n1(x))   # residual 1 (pre-norm attention)
        x = x + self.ff(self.n2(x))     # residual 2 (pre-norm feed-forward)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, n_feat, cfg):
        super().__init__()
        dm = cfg['d_model']
        self.se     = SqueezeExcite(n_feat)
        self.embed  = PatchEmbed(n_feat, cfg['patch_size'], dm)
        self.blocks = nn.ModuleList([
            TransformerBlock(dm, cfg['n_heads'], cfg['d_ff'], cfg['drop'])
            for _ in range(cfg['n_layers'])
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dm),
            nn.Linear(dm, dm // 2), nn.GELU(), nn.Dropout(cfg['drop']),
            nn.Linear(dm // 2, 2),
        )
    def forward(self, x):
        x = self.se(x); tok = self.embed(x)
        for blk in self.blocks: tok = blk(tok)
        return self.head(tok.mean(1))


# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════
def load_csv(path):
    wanted = {'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Number of trades', 'Quote asset volume'}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    df = df.rename(columns={
        'Open time': 'timestamp', 'Taker buy base asset volume': 'taker_buy_vol',
        'Number of trades': 'n_trades', 'Quote asset volume': 'quote_vol',
    })
    df['timestamp'] = pd.to_datetime(
        df['timestamp'].astype(str).str.strip().str.replace(' UTC', '', regex=False),
        utc=True, errors='coerce',
    )
    df = df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)


def atr_ema(h, l, c, period=14):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def add_indicators(df, px=''):
    d = df.copy()
    c, h, l, v, o = d['Close'], d['High'], d['Low'], d['Volume'], d['Open']
    atr14 = atr_ema(h, l, c, 14); atr7 = atr_ema(h, l, c, 7); ret1 = c.pct_change()

    if 'taker_buy_vol' in d.columns:
        buy_vol = d['taker_buy_vol']; sell_vol = v - buy_vol
        buy_r = buy_vol / (v + 1e-9); ofi = buy_vol - sell_vol
        d[f'{px}buy_ratio'] = buy_r; d[f'{px}buy_ratio_ma'] = buy_r.rolling(10).mean()
        d[f'{px}buy_ratio_dev'] = buy_r - buy_r.rolling(20).mean()
        d[f'{px}delta_vol'] = 2*buy_r - 1; d[f'{px}delta_vol_ma'] = d[f'{px}delta_vol'].rolling(10).mean()
        for w in [5, 10, 20]: d[f'{px}ofi{w}'] = ofi.rolling(w).sum() / (v.rolling(w).sum() + 1e-9)
        d[f'{px}ofi_mom'] = d[f'{px}ofi5'] - d[f'{px}ofi20']
        for w in [5, 20]: d[f'{px}vpin{w}'] = ofi.abs().rolling(w).sum() / (v.rolling(w).sum() + 1e-9)
        d[f'{px}ofi_div'] = (np.sign(ret1) - np.sign(ofi)).rolling(10).mean()
        if 'quote_vol' in d.columns:
            amihud = ret1.abs() / (d['quote_vol'] + 1e-9)
            d[f'{px}amihud'] = amihud / (amihud.rolling(20).mean() + 1e-9)
    if 'n_trades' in d.columns:
        nt = d['n_trades']; d[f'{px}trade_int'] = nt / (nt.rolling(20).mean() + 1e-9)
        avg_t = v / (nt + 1e-9); d[f'{px}trade_size'] = avg_t / (avg_t.rolling(20).mean() + 1e-9)

    vol_sync = (h - l) / (v + 1e-9); d[f'{px}vol_sync'] = vol_sync / (vol_sync.rolling(20).mean() + 1e-9)
    rv = (ret1**2).rolling(20).sum(); rbv = (ret1.abs()*ret1.shift(1).abs()).rolling(20).sum()*(np.pi/2)
    d[f'{px}jump_ratio'] = rv / (rbv + 1e-9)
    vsma = v.rolling(20).mean(); vr = v / (vsma + 1e-9)
    uw = (h - c.clip(lower=o)) / (atr14 + 1e-9); lw = (c.clip(upper=o) - l) / (atr14 + 1e-9)
    d[f'{px}upper_sweep'] = uw*vr; d[f'{px}lower_sweep'] = lw*vr
    d[f'{px}range_eff'] = ret1.abs() / ((h - l) / c + 1e-9)

    ema9  = c.ewm(span=9,  min_periods=9).mean()
    ema21 = c.ewm(span=21, min_periods=21).mean()
    ema50 = c.ewm(span=50, min_periods=50).mean()
    ema200= c.ewm(span=200,min_periods=200).mean()
    d[f'{px}pr9']=c/ema9-1; d[f'{px}pr21']=c/ema21-1; d[f'{px}pr50']=c/ema50-1; d[f'{px}pr200']=c/ema200-1
    d[f'{px}ema9_21']=ema9/ema21-1; d[f'{px}ema21_50']=ema21/ema50-1; d[f'{px}ema50_200']=ema50/ema200-1

    delta = c.diff()
    for per, name in [(7, 'rsi7'), (14, 'rsi14')]:
        g  = delta.clip(lower=0).ewm(alpha=1/per, min_periods=per).mean()
        ls = (-delta.clip(upper=0)).ewm(alpha=1/per, min_periods=per).mean()
        d[f'{px}{name}'] = (100 - 100 / (1 + g / (ls + 1e-9))) / 100
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    macd  = (ema12 - ema26) / (atr14 + 1e-9); macds = macd.ewm(span=9).mean()
    d[f'{px}macd']=macd; d[f'{px}macds']=macds; d[f'{px}macdh']=macd-macds
    d[f'{px}atr14']=atr14/(c+1e-9); d[f'{px}atr7']=atr7/(c+1e-9); d[f'{px}atr_r']=atr7/(atr14+1e-9)
    bm=c.rolling(20).mean(); bstd=c.rolling(20).std(); bup=bm+2*bstd; bdn=bm-2*bstd
    d[f'{px}bbw']=(bup-bdn)/(bm+1e-9); d[f'{px}bbp']=(c-bdn)/(bup-bdn+1e-9)
    ll14=l.rolling(14).min(); hh14=h.rolling(14).max()
    k=(c-ll14)/(hh14-ll14+1e-9); d[f'{px}stk']=k; d[f'{px}std']=k.rolling(3).mean()
    d[f'{px}wpr']=(hh14-c)/(hh14-ll14+1e-9)
    tp=(h+l+c)/3; tp_sma=tp.rolling(20).mean()
    tp_mad=tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    d[f'{px}cci']=(tp-tp_sma)/(0.015*tp_mad+1e-9)/200
    tenkan=(h.rolling(9).max()+l.rolling(9).min())/2; kijun=(h.rolling(26).max()+l.rolling(26).min())/2
    senA=((tenkan+kijun)/2).shift(26); senB=((h.rolling(52).max()+l.rolling(52).min())/2).shift(26)
    cloud_mid=(senA+senB)/2
    d[f'{px}ichi_tk']=(tenkan-kijun)/(c+1e-9); d[f'{px}ichi_pos']=(c-cloud_mid)/(c+1e-9); d[f'{px}ichi_cld']=(senA-senB)/(c+1e-9)
    vsma2=v.rolling(20).mean(); d[f'{px}vr']=v/(vsma2+1e-9); d[f'{px}vr5']=v.rolling(5).mean()/(vsma2+1e-9)
    obv=(np.sign(c.diff())*v).fillna(0).cumsum(); obv_m=obv.rolling(30).mean(); obv_s=obv.rolling(30).std()
    d[f'{px}obv']=(obv-obv_m)/(obv_s+1e-9)
    for n in [1,2,3,6,12,24]: d[f'{px}ret{n}']=c.pct_change(n)
    for n in [10,20]: d[f'{px}rvol{n}']=ret1.rolling(n).std()
    d[f'{px}skew20']=ret1.rolling(20).skew(); d[f'{px}kurt20']=ret1.rolling(20).kurt()
    d[f'{px}skew10']=ret1.rolling(10).skew(); d[f'{px}autocorr']=ret1.rolling(20).corr(ret1.shift(1))
    d[f'{px}body']=(c-o)/(atr14+1e-9); d[f'{px}hl']=(h-l)/(c+1e-9)
    d[f'{px}upper']=(h-c.clip(lower=o))/(atr14+1e-9); d[f'{px}lower']=(c.clip(upper=o)-l)/(atr14+1e-9)
    return d


def ind_cols(df, prefix):
    return ['timestamp'] + [c for c in df.columns if c.startswith(prefix)]


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Resolve prediction window ─────────────────────────────────
    if args.date_from:
        ts_from = parse_dt(args.date_from)
        ts_to   = parse_dt(args.date_to) if args.date_to else None  # None = latest
        label   = f"{ts_from.strftime('%Y-%m-%d %H:%M')}"
        label  += f"  ->  {ts_to.strftime('%Y-%m-%d %H:%M')}" if ts_to else "  ->  latest"
    else:
        ts_to   = None  # latest
        ts_from = None  # resolved after data load (latest minus --days)
        label   = f"last {args.days:.4g} day(s)"

    # ── Load checkpoint ───────────────────────────────────────────
    print(f"Device: {device}")
    print("Loading checkpoint...")
    ckpt      = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    model_cfg = ckpt['model_cfg']
    scaler    = ckpt['scaler']
    feat_cols = ckpt['feat_cols']
    seq_len   = ckpt['seq_len']
    vol_p33   = ckpt.get('vol_p33', None)
    vol_p67   = ckpt.get('vol_p67', None)
    print(f"  seq_len={seq_len}  n_feat={model_cfg['n_feat']}  "
          f"d_model={model_cfg['d_model']}  layers={model_cfg['n_layers']}")

    model = TemporalTransformer(model_cfg['n_feat'], model_cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load historical CSVs ──────────────────────────────────────
    print("\nLoading data...")
    df_4h  = load_csv(os.path.join(DATA_DIR, HIST_FILES['4h']))
    df_1h  = load_csv(os.path.join(DATA_DIR, HIST_FILES['1h']))
    df_15m = load_csv(os.path.join(DATA_DIR, HIST_FILES['15m']))
    df_1d  = load_csv(os.path.join(DATA_DIR, HIST_FILES['1d']))
    latest_ts = df_4h['timestamp'].iloc[-1]
    print(f"  4h:{len(df_4h):,}  1h:{len(df_1h):,}  15m:{len(df_15m):,}  1d:{len(df_1d):,}")
    print(f"  Latest 4h candle: {latest_ts.strftime('%Y-%m-%d %H:%M UTC')}")

    # Resolve --days relative to latest available candle
    if ts_from is None:
        ts_from = latest_ts - pd.Timedelta(hours=args.days * 24)

    print(f"  Prediction window: {label}")

    # ── Indicators ────────────────────────────────────────────────
    print("\nComputing indicators...")
    df_4h_f  = add_indicators(df_4h,  '4h_')
    df_1h_f  = add_indicators(df_1h,  '1h_')
    df_15m_f = add_indicators(df_15m, '15m_')
    df_1d_f  = add_indicators(df_1d,  '1d_')

    base = df_4h_f.copy()
    for other, px in [(df_1h_f, '1h_'), (df_1d_f, '1d_'), (df_15m_f, '15m_')]:
        base = pd.merge_asof(
            base.sort_values('timestamp'),
            other[ind_cols(other, px)].sort_values('timestamp'),
            on='timestamp', direction='backward',
        )
    base = base.sort_values('timestamp').reset_index(drop=True)

    for a, b, name in [('4h_rsi14','1d_rsi14','div_rsi_4h1d'),
                       ('4h_rsi14','1h_rsi14','div_rsi_4h1h'),
                       ('4h_macdh','1d_macdh','div_macdh_4h1d')]:
        if a in base.columns and b in base.columns:
            base[name] = base[a] - base[b]
    if '4h_vr' in base.columns and '1d_vr' in base.columns:
        base['div_vol_4h1d'] = base['4h_vr'] / (base['1d_vr'] + 1e-9)

    base = base.dropna(subset=feat_cols).reset_index(drop=True)

    # ── Scale (full history — DO NOT refit) ───────────────────────
    timestamps_all = base['timestamp'].values
    X_all          = base[feat_cols].values.astype(np.float32)
    X_scaled_all   = np.clip(scaler.transform(X_all), -6, 6).astype(np.float32)
    print(f"  Feature matrix: {X_all.shape}")

    # ── ATR regime boundaries ─────────────────────────────────────
    # Prefer the training-set boundaries saved in the checkpoint so the
    # regime split matches btc_lstm_train.py exactly. Fall back to
    # full-history percentiles for older checkpoints without them.
    vol_idx = feat_cols.index(VOL_FEATURE) if VOL_FEATURE in feat_cols else None
    if vol_idx is not None:
        if vol_p33 is not None and vol_p67 is not None:
            p33, p67 = vol_p33, vol_p67
        else:
            p33, p67 = np.percentile(X_scaled_all[:, vol_idx], [33, 67])

    # ── Resolve window indices ─────────────────────────────────────
    ts_series = pd.Series(pd.to_datetime(timestamps_all))   # tz-naive UTC from .values
    ts_from_naive = ts_from.tz_localize(None) if ts_from.tzinfo else ts_from
    ts_to_naive   = ts_to.tz_localize(None)   if (ts_to and ts_to.tzinfo) else ts_to

    # First index we want a prediction FOR
    from_idx = int((ts_series >= ts_from_naive).idxmax()) if (ts_series >= ts_from_naive).any() else len(base)
    # Last index we want a prediction FOR (inclusive)
    to_idx   = int((ts_series <= ts_to_naive).values.nonzero()[0][-1]) if ts_to_naive is not None else len(base) - 1

    if from_idx > to_idx:
        print(f"\nNo data found in requested window ({label}).")
        return

    # Trim to just enough history to build sequences for the window
    trim_start = max(0, from_idx - seq_len)
    X_trim  = X_scaled_all[trim_start:]
    ts_trim = timestamps_all[trim_start:]

    # Relative indices within the trim
    rel_from = from_idx - trim_start
    rel_to   = to_idx   - trim_start

    # Each prediction at position i needs X_trim[i-seq_len : i]
    # So the first sequence we build is at i = seq_len (gives prediction for trim index seq_len-1)
    # We need predictions for trim indices rel_from .. rel_to
    seq_indices = range(
        max(seq_len, rel_from + 1),   # first sequence output index (inclusive)
        rel_to + 2,                   # last  sequence output index (exclusive)
    )
    if not seq_indices:
        print(f"\nNot enough history to build sequences for the requested window.")
        print(f"Need at least {seq_len} candles before {ts_from_naive.strftime('%Y-%m-%d %H:%M')}.")
        return

    sequences = np.stack([X_trim[i - seq_len:i] for i in seq_indices])
    ts_seq    = ts_trim[[i - 1 for i in seq_indices]]
    print(f"  Sequences to run: {sequences.shape[0]}  (window: {label})")

    # ── Inference ─────────────────────────────────────────────────
    print("\nRunning inference...")
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(sequences), 128):
            xb   = torch.from_numpy(sequences[i:i+128]).to(device)
            prbs = torch.softmax(model(xb), dim=1).cpu().numpy()
            all_probs.append(prbs)
    probs  = np.concatenate(all_probs, axis=0)
    p_long = probs[:, 1]

    # ── Adaptive volatility thresholding ──────────────────────────
    if vol_idx is not None:
        vol_seq   = X_trim[[i - 1 for i in seq_indices], vol_idx]
        long_thr  = np.where(vol_seq > p67, THRESH_HIGH_VOL[0],
                    np.where(vol_seq < p33, THRESH_LOW_VOL[0], LONG_THRESH))
        short_thr = np.where(vol_seq > p67, THRESH_HIGH_VOL[1],
                    np.where(vol_seq < p33, THRESH_LOW_VOL[1], SHORT_THRESH))
        regime    = np.where(vol_seq > p67, 'high', np.where(vol_seq < p33, 'low', 'mid'))
    else:
        long_thr  = np.full(len(p_long), LONG_THRESH)
        short_thr = np.full(len(p_long), SHORT_THRESH)
        regime    = np.full(len(p_long), 'mid')

    sig = np.where(p_long >= long_thr, 'LONG',
         np.where(p_long <= short_thr, 'SHORT', 'NEUTRAL'))

    # ── Display ───────────────────────────────────────────────────
    n_long    = (sig == 'LONG').sum()
    n_short   = (sig == 'SHORT').sum()
    n_neutral = (sig == 'NEUTRAL').sum()

    print(f"\n{'='*62}")
    print(f"SIGNALS  —  {label}  ({len(sig)} candles)")
    print(f"{'='*62}")
    print(f"  LONG   : {n_long}    SHORT  : {n_short}    "
          f"NEUTRAL: {n_neutral}  ({100*n_neutral/max(len(sig),1):.1f}% filtered)")
    print()
    print(f"  {'Timestamp':<22}  {'p_long':>7}  {'p_short':>7}  {'Signal':<8}  "
          f"{'Vol':>4}  {'L thr':>5}  {'S thr':>5}")
    print(f"  {'─'*68}")
    for ts, pl, ps, s, reg, lt, st in zip(
            ts_seq, p_long, probs[:, 0], sig, regime, long_thr, short_thr):
        ts_str = pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')
        mark   = '  *' if s != 'NEUTRAL' else ''
        print(f"  {ts_str:<22}  {pl:>7.4f}  {ps:>7.4f}  {s:<8}  "
              f"{reg:>4}  {lt:>5.2f}  {st:>5.2f}{mark}")

    # ── Save ──────────────────────────────────────────────────────
    out = pd.DataFrame({
        'timestamp':  ts_seq,
        'prob_long':  p_long,
        'prob_short': probs[:, 0],
        'signal':     sig,
        'vol_regime': regime,
        'long_thr':   long_thr,
        'short_thr':  short_thr,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Active signals: {n_long + n_short} / {len(sig)}")


if __name__ == '__main__':
    main()
