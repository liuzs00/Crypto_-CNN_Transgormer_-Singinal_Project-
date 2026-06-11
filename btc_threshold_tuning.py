"""
btc_threshold_tuning.py — Calibrate vol-regime boundaries and LONG/SHORT thresholds

Why: The default thresholds in btc_predict.py (0.60/0.40 mid, 0.55/0.45 low, 0.72/0.28 high)
were set by intuition. This script finds optimal values from data on the held-out test set.

Tuning plan (3 stages):
  Stage A  —  Regime boundaries (25 runs)
               Grid: p_low ∈ {15,20,25,30,33}  ×  p_high ∈ {67,70,75,80,85}
               Finds the ATR percentile cutoffs that best separate vol regimes.
               Fixed thresholds during this stage; locked after.

  Stage B  —  Per-regime threshold search (243 runs = 81 per regime × 3 regimes)
               For each of low / mid / high vol independently:
               long_thr ∈ 9 values  ×  short_thr ∈ 9 values
               Constraint: long_thr > 0.50, short_thr < 0.50, coverage >= MIN_COVERAGE
               Metric: balanced_prec = 0.4*long_prec + 0.6*short_prec

  Stage C  —  Joint fine-tune + test-set validation (≈30 runs)
               Neighbourhood search around Stage B winners.
               Validates on test set; checks if per-regime thresholds beat unified.

Primary metric  : balanced_prec (precision weighted 40% Long / 60% Short)
Secondary metric: expected_pnl  = prec * TP_PCT - (1-prec) * SL_PCT  per signal
Coverage floor  : MIN_COVERAGE = 0.03 (at least 3% of candles must signal)

Output: threshold_tuning_results.csv  +  recommended config printed to console
"""

import os, sys, warnings, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS & SHARED CONSTANTS  (must match btc_lstm_train.py / btc_predict.py)
# ─────────────────────────────────────────────
DATA_DIR  = r"D:\Document\LLLLLLLLLLLLL\DATA"
SAVE_DIR  = r"D:\Document\LLLLLLLLLLLLL"
CKPT_PATH = os.path.join(SAVE_DIR, 'btc_lstm_model.pth')
OUT_CSV   = os.path.join(SAVE_DIR, 'threshold_tuning_results.csv')

HIST_FILES = {
    '4h' : 'btc_4h_data_2018_to_2025.csv',
    '1h' : 'btc_1h_data_2018_to_2025.csv',
    '15m': 'btc_15m_data_2018_to_2025.csv',
    '1d' : 'btc_1d_data_2018_to_2025.csv',
}

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.85

TP_PCT     = 0.015
SL_PCT     = 0.03
TB_TIMEOUT = 4

VOL_FEATURE  = '4h_atr14'
MIN_COVERAGE = 0.03     # reject configs with < 3% signal rate
MIN_SIGNALS  = 20       # reject configs with < 20 signals per side (too noisy)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────
# STAGE GRIDS
# ─────────────────────────────────────────────
# Stage A — regime boundary percentiles
A_P_LOW  = [15, 20, 25, 30, 33]
A_P_HIGH = [67, 70, 75, 80, 85]

# Stage B — per-regime threshold candidates
B_LONG_THRS  = [0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72]
B_SHORT_THRS = [0.48, 0.45, 0.43, 0.40, 0.38, 0.35, 0.32, 0.30, 0.28]

# Stage C — neighbourhood half-width for fine-tune
C_DELTA = 0.02


# ═══════════════════════════════════════════════════════════════════
# MODEL  (identical to btc_predict.py — must stay in sync)
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
        emb = torch.cat([freqs, freqs], dim=-1)
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
        return x + self.ff(self.n2(x + self.attn(self.n1(x))))

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
# DATA PIPELINE  (identical to btc_predict.py — must stay in sync)
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
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


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
        for w in [5,10,20]: d[f'{px}ofi{w}'] = ofi.rolling(w).sum()/(v.rolling(w).sum()+1e-9)
        d[f'{px}ofi_mom'] = d[f'{px}ofi5'] - d[f'{px}ofi20']
        for w in [5,20]: d[f'{px}vpin{w}'] = ofi.abs().rolling(w).sum()/(v.rolling(w).sum()+1e-9)
        d[f'{px}ofi_div'] = (np.sign(ret1) - np.sign(ofi)).rolling(10).mean()
        if 'quote_vol' in d.columns:
            amihud = ret1.abs()/(d['quote_vol']+1e-9)
            d[f'{px}amihud'] = amihud/(amihud.rolling(20).mean()+1e-9)
    if 'n_trades' in d.columns:
        nt = d['n_trades']; d[f'{px}trade_int'] = nt/(nt.rolling(20).mean()+1e-9)
        avg_t = v/(nt+1e-9); d[f'{px}trade_size'] = avg_t/(avg_t.rolling(20).mean()+1e-9)
    vol_sync = (h-l)/(v+1e-9); d[f'{px}vol_sync'] = vol_sync/(vol_sync.rolling(20).mean()+1e-9)
    rv = (ret1**2).rolling(20).sum(); rbv = (ret1.abs()*ret1.shift(1).abs()).rolling(20).sum()*(np.pi/2)
    d[f'{px}jump_ratio'] = rv/(rbv+1e-9)
    vsma = v.rolling(20).mean(); vr = v/(vsma+1e-9)
    uw = (h-c.clip(lower=o))/(atr14+1e-9); lw = (c.clip(upper=o)-l)/(atr14+1e-9)
    d[f'{px}upper_sweep'] = uw*vr; d[f'{px}lower_sweep'] = lw*vr
    d[f'{px}range_eff'] = ret1.abs()/((h-l)/c+1e-9)
    ema9=c.ewm(span=9,min_periods=9).mean(); ema21=c.ewm(span=21,min_periods=21).mean()
    ema50=c.ewm(span=50,min_periods=50).mean(); ema200=c.ewm(span=200,min_periods=200).mean()
    d[f'{px}pr9']=c/ema9-1; d[f'{px}pr21']=c/ema21-1; d[f'{px}pr50']=c/ema50-1; d[f'{px}pr200']=c/ema200-1
    d[f'{px}ema9_21']=ema9/ema21-1; d[f'{px}ema21_50']=ema21/ema50-1; d[f'{px}ema50_200']=ema50/ema200-1
    delta = c.diff()
    for per, name in [(7,'rsi7'),(14,'rsi14')]:
        g = delta.clip(lower=0).ewm(alpha=1/per,min_periods=per).mean()
        ls = (-delta.clip(upper=0)).ewm(alpha=1/per,min_periods=per).mean()
        d[f'{px}{name}'] = (100-100/(1+g/(ls+1e-9)))/100
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean()
    macd=(ema12-ema26)/(atr14+1e-9); macds=macd.ewm(span=9).mean()
    d[f'{px}macd']=macd; d[f'{px}macds']=macds; d[f'{px}macdh']=macd-macds
    d[f'{px}atr14']=atr14/(c+1e-9); d[f'{px}atr7']=atr7/(c+1e-9); d[f'{px}atr_r']=atr7/(atr14+1e-9)
    bm=c.rolling(20).mean(); bstd=c.rolling(20).std(); bup=bm+2*bstd; bdn=bm-2*bstd
    d[f'{px}bbw']=(bup-bdn)/(bm+1e-9); d[f'{px}bbp']=(c-bdn)/(bup-bdn+1e-9)
    ll14=l.rolling(14).min(); hh14=h.rolling(14).max()
    k=(c-ll14)/(hh14-ll14+1e-9); d[f'{px}stk']=k; d[f'{px}std']=k.rolling(3).mean()
    d[f'{px}wpr']=(hh14-c)/(hh14-ll14+1e-9)
    tp=(h+l+c)/3; tp_sma=tp.rolling(20).mean()
    tp_mad=tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())),raw=True)
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
# TRIPLE BARRIER LABELING  (matches btc_lstm_train.py exactly)
# ═══════════════════════════════════════════════════════════════════
def triple_barrier(close_arr, high_arr, low_arr, tp_pct, sl_pct, timeout):
    """
    label=1  Long:  TP hit first (High[t+k] >= close[t]*(1+tp_pct))
    label=0  Short: SL hit first (Low[t+k]  <= close[t]*(1-sl_pct))
    NaN      both barriers on same candle (ambiguous) OR timeout reached
    """
    n = len(close_arr)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        ref    = close_arr[i]
        tp_lvl = ref * (1 + tp_pct)
        sl_lvl = ref * (1 - sl_pct)
        for k in range(1, min(timeout + 1, n - i)):
            tp_hit = high_arr[i + k] >= tp_lvl
            sl_hit = low_arr[i + k]  <= sl_lvl
            if tp_hit and sl_hit:
                break               # ambiguous — leave as NaN
            elif tp_hit:
                labels[i] = 1; break
            elif sl_hit:
                labels[i] = 0; break
    return labels


# ═══════════════════════════════════════════════════════════════════
# THRESHOLD EVALUATOR  (pure numpy — fast for grid search)
# ═══════════════════════════════════════════════════════════════════
def evaluate(p_long, labels, atr_vol,
             p_low_pct, p_high_pct,
             lt_low, st_low, lt_mid, st_mid, lt_high, st_high,
             atr_all):
    """
    Returns a result dict or None if coverage / signal-count constraints not met.
    atr_all : full dataset ATR values (for computing percentile thresholds from training distribution)
    """
    p33 = np.percentile(atr_all, p_low_pct)
    p67 = np.percentile(atr_all, p_high_pct)

    regime = np.where(atr_vol > p67, 2,
             np.where(atr_vol < p33, 0, 1))          # 0=low 1=mid 2=high

    lt = np.where(regime == 2, lt_high, np.where(regime == 0, lt_low, lt_mid))
    st = np.where(regime == 2, st_high, np.where(regime == 0, st_low, st_mid))

    is_long  = p_long >= lt
    is_short = p_long <= st

    coverage = (is_long | is_short).mean()
    if coverage < MIN_COVERAGE:
        return None

    n_long  = is_long.sum()
    n_short = is_short.sum()
    if n_long < MIN_SIGNALS or n_short < MIN_SIGNALS:
        return None

    lp = labels[is_long].mean()           # long precision  = P(label=1 | signaled LONG)
    sp = (1 - labels[is_short]).mean()    # short precision = P(label=0 | signaled SHORT)

    bp = 0.4 * lp + 0.6 * sp

    return {
        'balanced_prec': round(float(bp), 4),
        'long_prec':     round(float(lp), 4),
        'short_prec':    round(float(sp), 4),
        'coverage':      round(float(coverage), 4),
        'n_long':        int(n_long),
        'n_short':       int(n_short),
        'exp_pnl_long':  round(float(lp  * TP_PCT - (1-lp)  * SL_PCT), 5),
        'exp_pnl_short': round(float(sp  * TP_PCT - (1-sp)  * SL_PCT), 5),
        'p_low_pct':     p_low_pct,  'p_high_pct':  p_high_pct,
        'lt_low':  lt_low,  'st_low':  st_low,
        'lt_mid':  lt_mid,  'st_mid':  st_mid,
        'lt_high': lt_high, 'st_high': st_high,
    }


def best(results):
    """Return the result dict with highest balanced_prec."""
    valid = [r for r in results if r is not None]
    return max(valid, key=lambda r: r['balanced_prec']) if valid else None


# ═══════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ═══════════════════════════════════════════════════════════════════
def run_inference(model, X_scaled, seq_len, batch=256):
    """Build sequences and run model; returns p_long array (N,)."""
    seqs = np.stack([X_scaled[i - seq_len:i]
                     for i in range(seq_len, len(X_scaled) + 1)])
    all_p = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            xb = torch.from_numpy(seqs[i:i+batch]).to(device)
            p  = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            all_p.append(p)
    return np.concatenate(all_p)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    # ── 1. Load checkpoint ───────────────────────────────────────
    print(f"Device: {device}")
    print("Loading checkpoint...")
    ckpt      = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    model_cfg = ckpt['model_cfg']
    scaler    = ckpt['scaler']
    feat_cols = ckpt['feat_cols']
    seq_len   = ckpt['seq_len']
    print(f"  seq_len={seq_len}  n_feat={model_cfg['n_feat']}  layers={model_cfg['n_layers']}")

    model = TemporalTransformer(model_cfg['n_feat'], model_cfg).to(device)
    model.load_state_dict(ckpt['model_state'])

    # ── 2. Load & build feature matrix ──────────────────────────
    print("\nLoading data and computing indicators...")
    df_4h  = load_csv(os.path.join(DATA_DIR, HIST_FILES['4h']))
    df_1h  = load_csv(os.path.join(DATA_DIR, HIST_FILES['1h']))
    df_15m = load_csv(os.path.join(DATA_DIR, HIST_FILES['15m']))
    df_1d  = load_csv(os.path.join(DATA_DIR, HIST_FILES['1d']))

    df_4h_f  = add_indicators(df_4h,  '4h_')
    df_1h_f  = add_indicators(df_1h,  '1h_')
    df_15m_f = add_indicators(df_15m, '15m_')
    df_1d_f  = add_indicators(df_1d,  '1d_')

    base = df_4h_f.copy()
    for other, px in [(df_1h_f,'1h_'), (df_1d_f,'1d_'), (df_15m_f,'15m_')]:
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

    # ── 3. Triple-barrier labels ──────────────────────────────────
    print("Generating triple-barrier labels...")
    base['label'] = triple_barrier(
        base['Close'].values, base['High'].values, base['Low'].values,
        TP_PCT, SL_PCT, TB_TIMEOUT,
    )
    base = base.dropna(subset=['label'] + feat_cols).reset_index(drop=True)
    base['label'] = base['label'].astype(int)
    n_total = len(base)
    n_long  = base['label'].sum()
    print(f"  Labeled rows: {n_total:,}  Long={n_long:,} ({100*n_long/n_total:.1f}%)  "
          f"Short={n_total-n_long:,} ({100*(n_total-n_long)/n_total:.1f}%)")

    # ── 4. Train / Val / Test split  (mirrors btc_lstm_train.py) ─
    X = base[feat_cols].values.astype(np.float32)
    y = base['label'].values.astype(np.int64)

    i_tr = int(n_total * TRAIN_FRAC)
    i_va = int(n_total * VAL_FRAC)

    X_tr = np.clip(scaler.transform(X[:i_tr]),  -6, 6).astype(np.float32)
    X_va = np.clip(scaler.transform(X[i_tr:i_va]), -6, 6).astype(np.float32)
    X_te = np.clip(scaler.transform(X[i_va:]),   -6, 6).astype(np.float32)
    y_va = y[i_tr:i_va]; y_te = y[i_va:]

    print(f"  Split — train:{i_tr:,}  val:{i_va-i_tr:,}  test:{n_total-i_va:,}")

    # ATR column index (used for regime classification)
    vol_idx = feat_cols.index(VOL_FEATURE)
    atr_all = np.clip(scaler.transform(X)[:, vol_idx], -6, 6)  # full dataset for percentile calc

    # ── 5. Run inference on val + test ───────────────────────────
    print("\nRunning inference on val set...", end=" ", flush=True)
    p_va = run_inference(model, X_va, seq_len)
    # Align labels with sequences (first seq_len-1 rows have no sequence)
    yva_s   = y_va[seq_len - 1:]
    atr_va  = X_va[seq_len - 1:, vol_idx]
    print(f"{len(p_va):,} sequences")

    print("Running inference on test set...", end=" ", flush=True)
    p_te = run_inference(model, X_te, seq_len)
    yte_s   = y_te[seq_len - 1:]
    atr_te  = X_te[seq_len - 1:, vol_idx]
    print(f"{len(p_te):,} sequences")

    print(f"\n  Val  p_long distribution: min={p_va.min():.3f}  "
          f"mean={p_va.mean():.3f}  max={p_va.max():.3f}")
    print(f"  Test p_long distribution: min={p_te.min():.3f}  "
          f"mean={p_te.mean():.3f}  max={p_te.max():.3f}")

    # ══════════════════════════════════════════════════════════════
    # STAGE A — Regime boundary search (tune on val, 25 runs)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE A — Regime boundary search  ({len(A_P_LOW)*len(A_P_HIGH)} runs on val set)")
    print(f"{'='*60}")

    # Fixed baseline thresholds during stage A
    A_FIXED = dict(lt_low=0.55, st_low=0.45, lt_mid=0.60,
                   st_mid=0.40, lt_high=0.72, st_high=0.28)

    a_results = []
    for p_low, p_high in itertools.product(A_P_LOW, A_P_HIGH):
        if p_high - p_low < 20:
            continue   # don't allow mid-regime narrower than 20 pp
        r = evaluate(p_va, yva_s, atr_va, p_low, p_high,
                     A_FIXED['lt_low'],  A_FIXED['st_low'],
                     A_FIXED['lt_mid'],  A_FIXED['st_mid'],
                     A_FIXED['lt_high'], A_FIXED['st_high'],
                     atr_all)
        if r: a_results.append(r)

    best_a = best(a_results)
    if best_a is None:
        print("  No valid configs found in Stage A — using defaults (33/67)")
        best_p_low, best_p_high = 33, 67
    else:
        best_p_low, best_p_high = best_a['p_low_pct'], best_a['p_high_pct']
        print(f"  Best: p_low={best_p_low}  p_high={best_p_high}  "
              f"balanced_prec={best_a['balanced_prec']:.4f}  "
              f"coverage={best_a['coverage']:.3f}")

    # Print top 5
    top_a = sorted(a_results, key=lambda r: -r['balanced_prec'])[:5]
    print(f"\n  {'p_low':>5}  {'p_high':>6}  {'b_prec':>7}  {'l_prec':>7}  "
          f"{'s_prec':>7}  {'cov':>5}  {'n_L':>5}  {'n_S':>5}")
    print(f"  {'─'*55}")
    for r in top_a:
        mark = ' <-- best' if r == best_a else ''
        print(f"  {r['p_low_pct']:>5}  {r['p_high_pct']:>6}  {r['balanced_prec']:>7.4f}  "
              f"{r['long_prec']:>7.4f}  {r['short_prec']:>7.4f}  {r['coverage']:>5.3f}  "
              f"{r['n_long']:>5}  {r['n_short']:>5}{mark}")

    # ══════════════════════════════════════════════════════════════
    # STAGE B — Per-regime threshold search (tune on val, 243 runs)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE B — Per-regime threshold search  "
          f"({len(B_LONG_THRS)**2 * 3} runs on val set)")
    print(f"{'='*60}")
    print(f"  Regime boundaries locked: p_low={best_p_low}  p_high={best_p_high}\n")

    best_per_regime = {}

    for regime_name, regime_label in [('low', 0), ('mid', 1), ('high', 2)]:
        p33 = np.percentile(atr_all, best_p_low)
        p67 = np.percentile(atr_all, best_p_high)

        if regime_label == 0:
            mask = atr_va < p33
        elif regime_label == 2:
            mask = atr_va > p67
        else:
            mask = (atr_va >= p33) & (atr_va <= p67)

        n_regime = mask.sum()
        if n_regime < MIN_SIGNALS * 2:
            print(f"  [{regime_name}]  only {n_regime} val samples — using default thresholds")
            defaults = {'low': (0.55, 0.45), 'mid': (0.60, 0.40), 'high': (0.72, 0.28)}
            best_per_regime[regime_name] = defaults[regime_name]
            continue

        p_reg  = p_va[mask]
        y_reg  = yva_s[mask]

        b_results = []
        for lt, st in itertools.product(B_LONG_THRS, B_SHORT_THRS):
            if lt <= st:
                continue     # thresholds must not overlap
            is_long  = p_reg >= lt
            is_short = p_reg <= st
            cov = (is_long | is_short).mean()
            if cov < MIN_COVERAGE or is_long.sum() < MIN_SIGNALS or is_short.sum() < MIN_SIGNALS:
                continue
            lp = y_reg[is_long].mean()
            sp = (1 - y_reg[is_short]).mean()
            bp = 0.4 * lp + 0.6 * sp
            b_results.append({
                'bp': bp, 'lp': lp, 'sp': sp, 'cov': cov,
                'lt': lt, 'st': st,
                'n_l': int(is_long.sum()), 'n_s': int(is_short.sum()),
            })

        if not b_results:
            print(f"  [{regime_name}]  no valid threshold combos — using default")
            defaults = {'low': (0.55, 0.45), 'mid': (0.60, 0.40), 'high': (0.72, 0.28)}
            best_per_regime[regime_name] = defaults[regime_name]
            continue

        top_b = sorted(b_results, key=lambda r: -r['bp'])
        best_b = top_b[0]
        best_per_regime[regime_name] = (best_b['lt'], best_b['st'])

        print(f"  [{regime_name:>4} vol]  n={n_regime:,}  "
              f"best: long_thr={best_b['lt']:.2f}  short_thr={best_b['st']:.2f}  "
              f"balanced_prec={best_b['bp']:.4f}  coverage={best_b['cov']:.3f}  "
              f"(L:{best_b['n_l']} S:{best_b['n_s']})")
        print(f"           long_prec={best_b['lp']:.4f}  short_prec={best_b['sp']:.4f}  "
              f"exp_pnl_L={best_b['lp']*TP_PCT-(1-best_b['lp'])*SL_PCT:.4f}  "
              f"exp_pnl_S={best_b['sp']*TP_PCT-(1-best_b['sp'])*SL_PCT:.4f}")

        # Print top 5 for this regime
        print(f"           {'lt':>5}  {'st':>5}  {'b_prec':>7}  {'l_prec':>7}  "
              f"{'s_prec':>7}  {'cov':>5}")
        for r in top_b[:5]:
            print(f"           {r['lt']:>5.2f}  {r['st']:>5.2f}  {r['bp']:>7.4f}  "
                  f"{r['lp']:>7.4f}  {r['sp']:>7.4f}  {r['cov']:>5.3f}")
        print()

    lt_low,  st_low  = best_per_regime.get('low',  (0.55, 0.45))
    lt_mid,  st_mid  = best_per_regime.get('mid',  (0.60, 0.40))
    lt_high, st_high = best_per_regime.get('high', (0.72, 0.28))

    # ══════════════════════════════════════════════════════════════
    # STAGE C — Joint fine-tune + test-set validation
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE C — Fine-tune + test-set validation")
    print(f"{'='*60}")

    def _neighbourhood(v, delta=C_DELTA, step=0.01):
        lo = round(v - delta, 2); hi = round(v + delta, 2)
        return [round(x, 2) for x in np.arange(lo, hi + step/2, step)]

    c_results = []
    for lt_l in _neighbourhood(lt_low):
        for st_l in _neighbourhood(st_low):
            if lt_l <= st_l: continue
            for lt_m in _neighbourhood(lt_mid):
                for st_m in _neighbourhood(st_mid):
                    if lt_m <= st_m: continue
                    for lt_h in _neighbourhood(lt_high):
                        for st_h in _neighbourhood(st_high):
                            if lt_h <= st_h: continue
                            r = evaluate(p_va, yva_s, atr_va,
                                         best_p_low, best_p_high,
                                         lt_l, st_l, lt_m, st_m, lt_h, st_h,
                                         atr_all)
                            if r: c_results.append(r)

    print(f"  Stage C combos evaluated: {len(c_results):,}")
    best_c = best(c_results) if c_results else None

    if best_c:
        lt_low,  st_low  = best_c['lt_low'],  best_c['st_low']
        lt_mid,  st_mid  = best_c['lt_mid'],  best_c['st_mid']
        lt_high, st_high = best_c['lt_high'], best_c['st_high']
        print(f"  Stage C val best: balanced_prec={best_c['balanced_prec']:.4f}  "
              f"coverage={best_c['coverage']:.3f}")

    # Validate final config on TEST SET
    print("\n  --- Test-set validation ---")
    r_test = evaluate(p_te, yte_s, atr_te,
                      best_p_low, best_p_high,
                      lt_low, st_low, lt_mid, st_mid, lt_high, st_high,
                      atr_all)

    # Also evaluate CURRENT (default) config on test set for comparison
    r_default = evaluate(p_te, yte_s, atr_te, 33, 67,
                         0.55, 0.45, 0.60, 0.40, 0.72, 0.28,
                         atr_all)

    if r_test:
        print(f"  TUNED  (test) : balanced_prec={r_test['balanced_prec']:.4f}  "
              f"long_prec={r_test['long_prec']:.4f}  short_prec={r_test['short_prec']:.4f}  "
              f"coverage={r_test['coverage']:.3f}  "
              f"(L:{r_test['n_long']} S:{r_test['n_short']})")
    if r_default:
        print(f"  DEFAULT(test) : balanced_prec={r_default['balanced_prec']:.4f}  "
              f"long_prec={r_default['long_prec']:.4f}  short_prec={r_default['short_prec']:.4f}  "
              f"coverage={r_default['coverage']:.3f}  "
              f"(L:{r_default['n_long']} S:{r_default['n_short']})")

    # ── Save all stage results ────────────────────────────────────
    all_rows = []
    for r in a_results:
        all_rows.append({'stage': 'A', 'set': 'val', **r})
    for r in c_results:
        all_rows.append({'stage': 'C', 'set': 'val', **r})
    if r_test:
        all_rows.append({'stage': 'final', 'set': 'test', **r_test})
    if r_default:
        all_rows.append({'stage': 'default', 'set': 'test', **r_default})

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"\nResults saved -> {OUT_CSV}")

    # ── Recommended config ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RECOMMENDED CONFIG for btc_predict.py")
    print(f"{'='*60}")
    print(f"  VOL_FEATURE     = '{VOL_FEATURE}'")
    print(f"  ATR_PLOW_PCT    = {best_p_low}   # percentile for low/mid boundary")
    print(f"  ATR_PHIGH_PCT   = {best_p_high}  # percentile for mid/high boundary")
    print(f"  LONG_THRESH     = {lt_mid:.2f}   # mid vol")
    print(f"  SHORT_THRESH    = {st_mid:.2f}  # mid vol")
    print(f"  THRESH_LOW_VOL  = ({lt_low:.2f}, {st_low:.2f})")
    print(f"  THRESH_HIGH_VOL = ({lt_high:.2f}, {st_high:.2f})")
    if r_test and r_default:
        delta = r_test['balanced_prec'] - r_default['balanced_prec']
        sign  = '+' if delta >= 0 else ''
        print(f"\n  Test set improvement vs default: {sign}{delta:.4f} balanced_prec")

if __name__ == '__main__':
    main()
