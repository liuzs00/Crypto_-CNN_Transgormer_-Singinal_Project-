"""
btc_lstm_tuning_r2.py — Round 2 hyperparameter tuning  (22 runs / 5 stages)

Builds on Round 1 best config (D=256, L=3, LR=1e-3, DROPOUT=0.25, TP/SL=1.5/3%).

Stage A (6 runs) — Short-precision: focal gamma x class-weight upscaling
Stage B (5 runs) — LR stability: warmup, patience, grad-clip, OneCycleLR
Stage C (3 runs) — Batch size (untested in R1)
Stage D (4 runs) — Augmentation intensity (untested in R1)
Stage E (4 runs) — LR x DROPOUT joint grid (greedy blind spot in R1)

New metric: balanced_prec = 0.4*long_prec + 0.6*short_prec
Stage A selects winner by balanced_prec; other stages by acc_active.
Results saved to tuning_results_r2.csv (separate from R1 file).
"""

import os, time, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

DATA_DIR    = r"D:\Document\LLLLLLLLLLLLL\DATA"
SAVE_DIR    = r"D:\Document\LLLLLLLLLLLLL"
RESULTS_CSV = os.path.join(SAVE_DIR, 'tuning_results_r2.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print("Round 2 tuning — 22 training runs across 5 stages.\n")


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════
def atr_ema(h, l, c, period=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


def add_indicators(df, px=''):
    d = df.copy()
    c, h, l, v, o = d['Close'], d['High'], d['Low'], d['Volume'], d['Open']
    atr14 = atr_ema(h, l, c, 14)
    atr7  = atr_ema(h, l, c,  7)
    ret1  = c.pct_change()

    if 'taker_buy_vol' in d.columns:
        buy_vol  = d['taker_buy_vol']
        sell_vol = v - buy_vol
        buy_r    = buy_vol / (v + 1e-9)
        ofi      = buy_vol - sell_vol
        d[f'{px}buy_ratio']     = buy_r
        d[f'{px}buy_ratio_ma']  = buy_r.rolling(10).mean()
        d[f'{px}buy_ratio_dev'] = buy_r - buy_r.rolling(20).mean()
        d[f'{px}delta_vol']     = 2 * buy_r - 1
        d[f'{px}delta_vol_ma']  = d[f'{px}delta_vol'].rolling(10).mean()
        for w in [5, 10, 20]:
            d[f'{px}ofi{w}'] = ofi.rolling(w).sum() / (v.rolling(w).sum() + 1e-9)
        d[f'{px}ofi_mom'] = d[f'{px}ofi5'] - d[f'{px}ofi20']
        for w in [5, 20]:
            d[f'{px}vpin{w}'] = ofi.abs().rolling(w).sum() / (v.rolling(w).sum() + 1e-9)
        price_dir = np.sign(ret1); ofi_dir = np.sign(ofi)
        d[f'{px}ofi_div'] = (price_dir - ofi_dir).rolling(10).mean()
        if 'quote_vol' in d.columns:
            amihud = ret1.abs() / (d['quote_vol'] + 1e-9)
            d[f'{px}amihud'] = amihud / (amihud.rolling(20).mean() + 1e-9)

    if 'n_trades' in d.columns:
        nt = d['n_trades']
        d[f'{px}trade_int']  = nt / (nt.rolling(20).mean() + 1e-9)
        avg_t = v / (nt + 1e-9)
        d[f'{px}trade_size'] = avg_t / (avg_t.rolling(20).mean() + 1e-9)

    vol_sync = (h - l) / (v + 1e-9)
    d[f'{px}vol_sync']   = vol_sync / (vol_sync.rolling(20).mean() + 1e-9)
    rv  = (ret1 ** 2).rolling(20).sum()
    rbv = (ret1.abs() * ret1.shift(1).abs()).rolling(20).sum() * (np.pi / 2)
    d[f'{px}jump_ratio'] = rv / (rbv + 1e-9)

    upper_wick = (h - c.clip(lower=o))  / (atr14 + 1e-9)
    lower_wick = (c.clip(upper=o) - l) / (atr14 + 1e-9)
    vsma_loc   = v.rolling(20).mean()
    vr_loc     = v / (vsma_loc + 1e-9)
    d[f'{px}upper_sweep'] = upper_wick * vr_loc
    d[f'{px}lower_sweep'] = lower_wick * vr_loc
    d[f'{px}range_eff']   = ret1.abs() / (((h - l) / c) + 1e-9)

    ema9   = c.ewm(span=9,   min_periods=9).mean()
    ema21  = c.ewm(span=21,  min_periods=21).mean()
    ema50  = c.ewm(span=50,  min_periods=50).mean()
    ema200 = c.ewm(span=200, min_periods=200).mean()
    d[f'{px}pr9']       = c / ema9   - 1
    d[f'{px}pr21']      = c / ema21  - 1
    d[f'{px}pr50']      = c / ema50  - 1
    d[f'{px}pr200']     = c / ema200 - 1
    d[f'{px}ema9_21']   = ema9  / ema21  - 1
    d[f'{px}ema21_50']  = ema21 / ema50  - 1
    d[f'{px}ema50_200'] = ema50 / ema200 - 1

    delta = c.diff()
    for per, name in [(7, 'rsi7'), (14, 'rsi14')]:
        g  = delta.clip(lower=0).ewm(alpha=1/per, min_periods=per).mean()
        ls = (-delta.clip(upper=0)).ewm(alpha=1/per, min_periods=per).mean()
        d[f'{px}{name}'] = (100 - 100 / (1 + g / (ls + 1e-9))) / 100

    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    macd  = (ema12 - ema26) / (atr14 + 1e-9); macds = macd.ewm(span=9).mean()
    d[f'{px}macd']  = macd; d[f'{px}macds'] = macds; d[f'{px}macdh'] = macd - macds

    d[f'{px}atr14'] = atr14 / (c + 1e-9)
    d[f'{px}atr7']  = atr7  / (c + 1e-9)
    d[f'{px}atr_r'] = atr7  / (atr14 + 1e-9)

    bm = c.rolling(20).mean(); bstd = c.rolling(20).std()
    bup = bm + 2*bstd; bdn = bm - 2*bstd
    d[f'{px}bbw'] = (bup - bdn) / (bm + 1e-9)
    d[f'{px}bbp'] = (c - bdn)   / (bup - bdn + 1e-9)

    ll14 = l.rolling(14).min(); hh14 = h.rolling(14).max()
    k = (c - ll14) / (hh14 - ll14 + 1e-9)
    d[f'{px}stk'] = k; d[f'{px}std'] = k.rolling(3).mean()
    d[f'{px}wpr'] = (hh14 - c) / (hh14 - ll14 + 1e-9)

    tp = (h + l + c) / 3; tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d[f'{px}cci'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-9) / 200

    tenkan = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senA   = ((tenkan + kijun) / 2).shift(26)
    senB   = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    cloud_mid = (senA + senB) / 2
    d[f'{px}ichi_tk']  = (tenkan - kijun) / (c + 1e-9)
    d[f'{px}ichi_pos'] = (c - cloud_mid)  / (c + 1e-9)
    d[f'{px}ichi_cld'] = (senA   - senB)  / (c + 1e-9)

    vsma2 = v.rolling(20).mean()
    d[f'{px}vr']  = v / (vsma2 + 1e-9)
    d[f'{px}vr5'] = v.rolling(5).mean() / (vsma2 + 1e-9)
    obv   = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_m = obv.rolling(30).mean(); obv_s = obv.rolling(30).std()
    d[f'{px}obv'] = (obv - obv_m) / (obv_s + 1e-9)

    for n in [1, 2, 3, 6, 12, 24]:
        d[f'{px}ret{n}'] = c.pct_change(n)
    for n in [10, 20]:
        d[f'{px}rvol{n}'] = ret1.rolling(n).std()
    d[f'{px}skew20']   = ret1.rolling(20).skew()
    d[f'{px}kurt20']   = ret1.rolling(20).kurt()
    d[f'{px}skew10']   = ret1.rolling(10).skew()
    d[f'{px}autocorr'] = ret1.rolling(20).corr(ret1.shift(1))

    d[f'{px}body']  = (c - o) / (atr14 + 1e-9)
    d[f'{px}hl']    = (h - l) / (c + 1e-9)
    d[f'{px}upper'] = (h - c.clip(lower=o))  / (atr14 + 1e-9)
    d[f'{px}lower'] = (c.clip(upper=o) - l) / (atr14 + 1e-9)
    return d


def triple_barrier(close_arr, high_arr, low_arr, tp_pct, sl_pct, timeout):
    n = len(close_arr); labels = np.full(n, np.nan)
    for i in range(n - 1):
        ref = close_arr[i]
        tp_lvl = ref * (1 + tp_pct); sl_lvl = ref * (1 - sl_pct)
        for j in range(i + 1, min(i + 1 + timeout, n)):
            tp_hit = high_arr[j] >= tp_lvl; sl_hit = low_arr[j] <= sl_lvl
            if tp_hit and sl_hit: break
            elif tp_hit: labels[i] = 1; break
            elif sl_hit: labels[i] = 0; break
    return labels


# ═══════════════════════════════════════════════════════════════════
# MODEL  (parameterised — no globals)
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
        Q, K     = _apply_rope(Q, K, cos, sin)
        logits   = (Q @ K.transpose(-2, -1)) * self.scl + _alibi_bias(self.H, L, x.device)
        attn     = self.drop(logits.softmax(dim=-1))
        return self.out((attn @ V).transpose(1, 2).reshape(B, L, D))


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
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, n_feat, cfg):
        super().__init__()
        dm = cfg['D_MODEL']
        self.se     = SqueezeExcite(n_feat)
        self.embed  = PatchEmbed(n_feat, cfg['PATCH_SIZE'], dm)
        self.blocks = nn.ModuleList([
            TransformerBlock(dm, cfg['N_HEADS'], cfg['D_FF'], cfg['DROPOUT'])
            for _ in range(cfg['N_LAYERS'])
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dm),
            nn.Linear(dm, dm // 2), nn.GELU(), nn.Dropout(cfg['DROPOUT']),
            nn.Linear(dm // 2, 2),
        )
    def forward(self, x):
        x = self.se(x); tok = self.embed(x)
        for blk in self.blocks: tok = blk(tok)
        return self.head(tok.mean(1))


class FocalLS(nn.Module):
    def __init__(self, gamma=2.0, smooth=0.1, weight=None):
        super().__init__()
        self.gamma = gamma; self.smooth = smooth; self.weight = weight
    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            y_sm = torch.full_like(logits, self.smooth / (n - 1))
            y_sm.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)
        log_p = F.log_softmax(logits, dim=1)
        ce    = -(y_sm * log_p).sum(1)
        pt    = torch.exp(-F.cross_entropy(logits, targets,
                                            weight=self.weight, reduction='none'))
        return (((1 - pt) ** self.gamma) * ce).mean()


class SeqDS(Dataset):
    def __init__(self, X, y, noise=0.0, mask_p=0.0):
        self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
        self.noise = noise; self.mask_p = mask_p
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.noise  > 0: x += self.noise * torch.randn_like(x)
        if self.mask_p > 0:
            m = torch.bernoulli(torch.full((x.shape[-1],), self.mask_p)).bool()
            x[:, m] = 0.0
        return x, self.y[i]


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING  (runs once before the sweep)
# ═══════════════════════════════════════════════════════════════════
def load_csv(path):
    wanted = {'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Number of trades', 'Quote asset volume'}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    df = df.rename(columns={
        'Open time': 'timestamp', 'Taker buy base asset volume': 'taker_buy_vol',
        'Number of trades': 'n_trades', 'Quote asset volume': 'quote_vol',
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col != 'timestamp': df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

print("Loading CSVs…")
df_15m = load_csv(os.path.join(DATA_DIR, 'btc_15m_data_2018_to_2025.csv'))
df_1h  = load_csv(os.path.join(DATA_DIR, 'btc_1h_data_2018_to_2025.csv'))
df_4h  = load_csv(os.path.join(DATA_DIR, 'btc_4h_data_2018_to_2025.csv'))
df_1d  = load_csv(os.path.join(DATA_DIR, 'btc_1d_data_2018_to_2025.csv'))

print("Computing indicators…")
df_15m_f = add_indicators(df_15m, '15m_')
df_1h_f  = add_indicators(df_1h,  '1h_')
df_4h_f  = add_indicators(df_4h,  '4h_')
df_1d_f  = add_indicators(df_1d,  '1d_')

def ind_cols(df, prefix):
    return ['timestamp'] + [c for c in df.columns if c.startswith(prefix)]

print("Merging timeframes…")
BASE = df_4h_f.copy()
for other, px in [(df_1h_f, '1h_'), (df_1d_f, '1d_'), (df_15m_f, '15m_')]:
    BASE = pd.merge_asof(
        BASE.sort_values('timestamp'),
        other[ind_cols(other, px)].sort_values('timestamp'),
        on='timestamp', direction='backward'
    )
BASE = BASE.sort_values('timestamp').reset_index(drop=True)

for a, b, name in [('4h_rsi14', '1d_rsi14', 'div_rsi_4h1d'),
                   ('4h_rsi14', '1h_rsi14', 'div_rsi_4h1h'),
                   ('4h_macdh', '1d_macdh', 'div_macdh_4h1d')]:
    if a in BASE.columns and b in BASE.columns:
        BASE[name] = BASE[a] - BASE[b]
if '4h_vr' in BASE.columns and '1d_vr' in BASE.columns:
    BASE['div_vol_4h1d'] = BASE['4h_vr'] / (BASE['1d_vr'] + 1e-9)

RAW_COLS  = {'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'label',
             'taker_buy_vol', 'n_trades', 'quote_vol'}
FEAT_COLS = [c for c in BASE.columns if c not in RAW_COLS and BASE[c].nunique() > 1]
print(f"Base shape: {BASE.shape}  |  Features: {len(FEAT_COLS)}\n")


# ═══════════════════════════════════════════════════════════════════
# BASELINE_R2  (Round 1 best config)
# ═══════════════════════════════════════════════════════════════════
BASELINE_R2 = dict(
    D_MODEL=256, N_HEADS=8, N_LAYERS=3, D_FF=512,
    PATCH_SIZE=4, SEQ_LEN=64,
    DROPOUT=0.25, LABEL_SMOOTH=0.10,
    LR_MAX=1e-3, WARMUP_EP=10, PATIENCE=20,
    BATCH=128, EPOCHS=100,
    AUG_NOISE=0.02, AUG_MASK_P=0.10,
    TP_PCT=0.015, SL_PCT=0.03, TB_TIMEOUT=4,
    LONG_THRESH=0.60, SHORT_THRESH=0.40,
    # New R2 parameters (defaults reproduce R1 behaviour exactly)
    FOCAL_GAMMA=2.0,
    SHORT_WEIGHT_FACTOR=1.0,   # multiplier on top of balanced class weights
    GRAD_CLIP=1.0,
    SCHEDULER_TYPE='cosine',   # 'cosine' | 'onecycle'
)


# ═══════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════
def make_seq(X, y, seq):
    Xs = np.stack([X[i - seq:i] for i in range(seq, len(X) + 1)])
    return Xs, y[seq - 1:]


def run_experiment(cfg, run_label, stage_name):
    t0 = time.time()
    print(f"\n{'─'*65}")
    print(f"[{run_label}]  stage={stage_name}", flush=True)

    tmp = BASE.copy()
    tmp['label'] = triple_barrier(
        tmp['Close'].values, tmp['High'].values, tmp['Low'].values,
        cfg['TP_PCT'], cfg['SL_PCT'], cfg['TB_TIMEOUT']
    )
    tmp = tmp.dropna(subset=['label'])
    tmp['label'] = tmp['label'].astype(int)
    tmp = tmp.dropna(subset=FEAT_COLS).reset_index(drop=True)

    X = tmp[FEAT_COLS].values.astype(np.float32)
    y = tmp['label'].values.astype(np.int64)
    n = len(X); i_tr = int(n * 0.70); i_va = int(n * 0.85)
    X_tr, y_tr = X[:i_tr],     y[:i_tr]
    X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
    X_te, y_te = X[i_va:],     y[i_va:]

    scaler = RobustScaler(quantile_range=(5, 95))
    X_tr = np.clip(scaler.fit_transform(X_tr), -6, 6).astype(np.float32)
    X_va = np.clip(scaler.transform(X_va),      -6, 6).astype(np.float32)
    X_te = np.clip(scaler.transform(X_te),      -6, 6).astype(np.float32)

    SEQ = cfg['SEQ_LEN']
    Xtr_s, ytr_s = make_seq(X_tr, y_tr, SEQ)
    Xva_s, yva_s = make_seq(X_va, y_va, SEQ)
    Xte_s, yte_s = make_seq(X_te, y_te, SEQ)
    counts = Counter(ytr_s.tolist()); total = sum(counts.values())

    BATCH = cfg['BATCH']
    train_dl = DataLoader(SeqDS(Xtr_s, ytr_s, cfg['AUG_NOISE'], cfg['AUG_MASK_P']),
                          batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    val_dl   = DataLoader(SeqDS(Xva_s, yva_s), batch_size=BATCH, shuffle=False, num_workers=0)
    test_dl  = DataLoader(SeqDS(Xte_s, yte_s), batch_size=BATCH, shuffle=False, num_workers=0)

    n_feat   = Xtr_s.shape[2]
    model    = TemporalTransformer(n_feat, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  train={Xtr_s.shape[0]:,}  "
          f"Long={counts[1]:,}  Short={counts[0]:,}", flush=True)

    # Balanced class weights + optional Short upscaling
    w_base = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32)
    w_mult = torch.tensor([cfg['SHORT_WEIGHT_FACTOR'], 1.0])
    w      = (w_base * w_mult).to(device)

    criterion = FocalLS(gamma=cfg['FOCAL_GAMMA'], smooth=cfg['LABEL_SMOOTH'], weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LR_MAX'], weight_decay=1e-3)

    if cfg['SCHEDULER_TYPE'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg['LR_MAX'],
            steps_per_epoch=len(train_dl),
            epochs=cfg['EPOCHS'],
            pct_start=cfg['WARMUP_EP'] / cfg['EPOCHS'],
        )
        per_step = True
    else:
        def lr_lambda(ep):
            if ep < cfg['WARMUP_EP']: return (ep + 1) / cfg['WARMUP_EP']
            p = (ep - cfg['WARMUP_EP']) / max(1, cfg['EPOCHS'] - cfg['WARMUP_EP'])
            return 0.5 * (1 + np.cos(np.pi * p))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        per_step  = False

    grad_clip  = cfg['GRAD_CLIP']
    best_val   = float('inf'); best_state = None; no_imp = 0; ep_stopped = cfg['EPOCHS']

    for ep in range(1, cfg['EPOCHS'] + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if per_step: scheduler.step()
        if not per_step: scheduler.step()

        model.eval()
        vl, vpreds, vtrue = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item()
                vpreds.extend(out.argmax(1).cpu().tolist())
                vtrue.extend(yb.cpu().tolist())
        vl /= len(val_dl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= cfg['PATIENCE']:
            ep_stopped = ep; break
        if ep % 10 == 0 or ep == 1:
            va = accuracy_score(vtrue, vpreds)
            print(f"    ep{ep:3d}  vl={vl:.4f}  va={va:.4f}", flush=True)

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    tpreds, tprobs, ttrue = [], [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            out  = model(xb.to(device))
            prbs = torch.softmax(out, 1).cpu().numpy()
            tpreds.extend(out.argmax(1).cpu().tolist())
            tprobs.extend(prbs.tolist())
            ttrue.extend(yb.tolist())

    tpreds = np.array(tpreds); tprobs = np.array(tprobs); ttrue = np.array(ttrue)
    p_long = tprobs[:, 1]
    sig    = np.where(p_long >= cfg['LONG_THRESH'],  1,
             np.where(p_long <= cfg['SHORT_THRESH'], 0, -1))
    mask   = sig != -1

    acc_am = accuracy_score(ttrue, tpreds)
    acc_ac = accuracy_score(ttrue[mask], sig[mask]) if mask.sum() > 0 else 0.0
    rpt    = classification_report(ttrue[mask], sig[mask],
                                    target_names=['Short', 'Long'],
                                    output_dict=True, zero_division=0) if mask.sum() > 0 else {}

    lp = round(rpt.get('Long',  {}).get('precision', 0), 3)
    lr = round(rpt.get('Long',  {}).get('recall',    0), 3)
    sp = round(rpt.get('Short', {}).get('precision', 0), 3)
    sr = round(rpt.get('Short', {}).get('recall',    0), 3)
    balanced_prec = round(0.4 * lp + 0.6 * sp, 4)

    elapsed = time.time() - t0
    result  = dict(
        run=run_label, stage=stage_name,
        val_loss=round(best_val, 5),
        acc_argmax=round(acc_am, 4),
        acc_active=round(acc_ac, 4),
        n_active=int(mask.sum()),
        pct_filtered=round(100 * (sig == -1).sum() / len(sig), 1),
        long_prec=lp, long_rec=lr,
        short_prec=sp, short_rec=sr,
        balanced_prec=balanced_prec,
        ep_stopped=ep_stopped, n_params=n_params,
        elapsed_min=round(elapsed / 60, 1),
        **cfg,
    )
    print(f"  => val_loss={result['val_loss']}  acc_active={result['acc_active']}  "
          f"balanced_prec={result['balanced_prec']}  short_prec={result['short_prec']}  "
          f"ep={result['ep_stopped']}  {result['elapsed_min']}min", flush=True)

    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
all_results = []

def save_results():
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

def pick_best(results):
    return max(results, key=lambda r: (r['acc_active'], -r['val_loss']))

def pick_best_balanced(results):
    return max(results, key=lambda r: (r['balanced_prec'], -r['val_loss']))

def print_stage_table(results, stage_name, varied_key, metric='acc_active'):
    best = pick_best_balanced(results) if metric == 'balanced_prec' else pick_best(results)
    print(f"\n{'Stage ' + stage_name + ' — ' + varied_key:─^65}")
    print(f"  {'run':<20}  {varied_key:<14}  {'val_loss':>9}  "
          f"{'acc_active':>10}  {'short_prec':>10}  {'balanced_prec':>13}  {'ep':>4}")
    for r in sorted(results, key=lambda x: x.get(metric, 0), reverse=True):
        tag = '  <-- best' if r is best else ''
        val = str(r.get(varied_key, '?'))
        print(f"  {r['run']:<20}  {val:<14}  {r['val_loss']:>9.5f}  "
              f"{r['acc_active']:>10.4f}  {r['short_prec']:>10.3f}  "
              f"{r['balanced_prec']:>13.4f}  {r['ep_stopped']:>4}{tag}")


# ═══════════════════════════════════════════════════════════════════
# STAGE A — Short Signal Precision  (6 runs)
# Winner selected by: balanced_prec = 0.4*long_prec + 0.6*short_prec
# R1 reference: short_prec=0.602  long_prec=0.824  balanced_prec=0.691
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("STAGE A — Short Signal Precision  (focal gamma + class weight)")
print("=" * 65)
print("R1 reference: short_prec=0.602  balanced_prec=0.691")
print("Selection metric: balanced_prec = 0.4*long_prec + 0.6*short_prec\n")

sa_variants = [
    ('A1-g1.5-w1.0', dict(FOCAL_GAMMA=1.5, SHORT_WEIGHT_FACTOR=1.0)),  # softer focus
    ('A2-g3.0-w1.0', dict(FOCAL_GAMMA=3.0, SHORT_WEIGHT_FACTOR=1.0)),  # harder focus
    ('A3-g2.0-w1.5', dict(FOCAL_GAMMA=2.0, SHORT_WEIGHT_FACTOR=1.5)),  # short boost x1.5
    ('A4-g2.0-w2.0', dict(FOCAL_GAMMA=2.0, SHORT_WEIGHT_FACTOR=2.0)),  # short boost x2.0
    ('A5-g3.0-w1.5', dict(FOCAL_GAMMA=3.0, SHORT_WEIGHT_FACTOR=1.5)),  # hard focus + boost
    ('A6-g1.5-w2.0', dict(FOCAL_GAMMA=1.5, SHORT_WEIGHT_FACTOR=2.0)),  # soft focus + boost
]
sa_results = []
for name, overrides in sa_variants:
    cfg = {**BASELINE_R2, **overrides}
    r   = run_experiment(cfg, name, 'Short-Precision')
    all_results.append(r); sa_results.append(r); save_results()

print_stage_table(sa_results, 'A', 'FOCAL_GAMMA', metric='balanced_prec')
best_sa    = pick_best_balanced(sa_results)
BEST_GAMMA = best_sa['FOCAL_GAMMA']
BEST_SWF   = best_sa['SHORT_WEIGHT_FACTOR']
print(f"\n  Locked: FOCAL_GAMMA={BEST_GAMMA}  SHORT_WEIGHT_FACTOR={BEST_SWF}")
BEST_A = dict(FOCAL_GAMMA=BEST_GAMMA, SHORT_WEIGHT_FACTOR=BEST_SWF)


# ═══════════════════════════════════════════════════════════════════
# STAGE B — LR Stability  (5 runs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE B — LR Stability")
print("=" * 65)

sb_variants = [
    ('B1-LR7e4',  dict(LR_MAX=7e-4)),
    ('B2-W20',    dict(LR_MAX=1e-3, WARMUP_EP=20)),
    ('B3-P30',    dict(LR_MAX=1e-3, PATIENCE=30)),
    ('B4-GC05',   dict(LR_MAX=1e-3, GRAD_CLIP=0.5)),
    ('B5-OC',     dict(LR_MAX=1e-3, SCHEDULER_TYPE='onecycle')),
]
sb_results = []
for name, overrides in sb_variants:
    cfg = {**BASELINE_R2, **BEST_A, **overrides}
    r   = run_experiment(cfg, name, 'LR-Stability')
    all_results.append(r); sb_results.append(r); save_results()

print_stage_table(sb_results, 'B', 'LR_MAX')
best_sb = pick_best(sb_results)
BEST_B  = dict(
    LR_MAX=best_sb['LR_MAX'],
    WARMUP_EP=best_sb['WARMUP_EP'],
    PATIENCE=best_sb['PATIENCE'],
    GRAD_CLIP=best_sb['GRAD_CLIP'],
    SCHEDULER_TYPE=best_sb['SCHEDULER_TYPE'],
)
print(f"\n  Locked: LR_MAX={BEST_B['LR_MAX']:.1e}  WARMUP={BEST_B['WARMUP_EP']}  "
      f"PATIENCE={BEST_B['PATIENCE']}  GRAD_CLIP={BEST_B['GRAD_CLIP']}  "
      f"SCHEDULER={BEST_B['SCHEDULER_TYPE']}")


# ═══════════════════════════════════════════════════════════════════
# STAGE C — Batch Size  (3 runs; never tested in R1)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE C — Batch Size  (never tested in R1)")
print("=" * 65)

sc_variants = [
    ('C1-B64',       dict(BATCH=64)),
    ('C2-B256',      dict(BATCH=256)),
    ('C3-B64-LR5e4', dict(BATCH=64, LR_MAX=5e-4)),   # linear LR scaling: halve batch -> halve LR
]
sc_results = []
for name, overrides in sc_variants:
    cfg = {**BASELINE_R2, **BEST_A, **BEST_B, **overrides}
    r   = run_experiment(cfg, name, 'Batch')
    all_results.append(r); sc_results.append(r); save_results()

# Include Stage B winner (BATCH=128) as reference
sb_ref = best_sb.copy(); sb_ref['run'] = 'SB-best(B128)'
sc_compare = sc_results + [sb_ref]
print_stage_table(sc_compare, 'C', 'BATCH')
best_sc    = pick_best(sc_compare)
BEST_BATCH = best_sc['BATCH']
# If C3 (scaled LR) wins, propagate its LR into BEST_B
if best_sc.get('run', '').startswith('C3'):
    BEST_B = {**BEST_B, 'LR_MAX': best_sc['LR_MAX']}
print(f"\n  Locked: BATCH={BEST_BATCH}")
BEST_C = dict(BATCH=BEST_BATCH)


# ═══════════════════════════════════════════════════════════════════
# STAGE D — Augmentation Intensity  (4 runs; never tested in R1)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE D — Augmentation Intensity  (never tested in R1)")
print("=" * 65)

sd_variants = [
    ('D1-N01-M10', dict(AUG_NOISE=0.01, AUG_MASK_P=0.10)),  # less jitter
    ('D2-N03-M15', dict(AUG_NOISE=0.03, AUG_MASK_P=0.15)),  # more noise + more mask
    ('D3-N02-M20', dict(AUG_NOISE=0.02, AUG_MASK_P=0.20)),  # more masking only
    ('D4-N00-M25', dict(AUG_NOISE=0.00, AUG_MASK_P=0.25)),  # masking only, no jitter
]
sd_results = []
for name, overrides in sd_variants:
    cfg = {**BASELINE_R2, **BEST_A, **BEST_B, **BEST_C, **overrides}
    r   = run_experiment(cfg, name, 'Augmentation')
    all_results.append(r); sd_results.append(r); save_results()

# Include Stage C winner as reference (AUG_NOISE=0.02, AUG_MASK_P=0.10)
sc_ref = best_sc.copy(); sc_ref['run'] = 'SC-best(N02-M10)'
sd_compare = sd_results + [sc_ref]
print_stage_table(sd_compare, 'D', 'AUG_NOISE')
best_sd    = pick_best(sd_compare)
BEST_NOISE = best_sd['AUG_NOISE']
BEST_MASKP = best_sd['AUG_MASK_P']
print(f"\n  Locked: AUG_NOISE={BEST_NOISE}  AUG_MASK_P={BEST_MASKP}")
BEST_D = dict(AUG_NOISE=BEST_NOISE, AUG_MASK_P=BEST_MASKP)


# ═══════════════════════════════════════════════════════════════════
# STAGE E — LR x DROPOUT Joint Grid  (4 runs)
# Tests whether R1 greedy path missed a better joint optimum.
# Uses A + B (non-LR parts) + C + D locks; explicitly varies LR & DROPOUT.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE E — LR x DROPOUT Joint Grid  (R1 greedy blind spot)")
print("=" * 65)

# Carry Stage B's schedule/clip settings but let LR and DROPOUT be free
B_non_lr_drop = {k: v for k, v in BEST_B.items()
                 if k not in ('LR_MAX', 'DROPOUT')}

se_variants = [
    ('E1-LR1e3-D20', dict(LR_MAX=1e-3, DROPOUT=0.20)),  # high LR, less dropout
    ('E2-LR7e4-D20', dict(LR_MAX=7e-4, DROPOUT=0.20)),  # moderate LR, less dropout
    ('E3-LR7e4-D30', dict(LR_MAX=7e-4, DROPOUT=0.30)),  # moderate LR, more dropout
    ('E4-LR5e4-D15', dict(LR_MAX=5e-4, DROPOUT=0.15)),  # lower LR, minimal dropout
]
se_results = []
for name, overrides in se_variants:
    cfg = {**BASELINE_R2, **BEST_A, **B_non_lr_drop, **BEST_C, **BEST_D, **overrides}
    r   = run_experiment(cfg, name, 'LR-x-Dropout')
    all_results.append(r); se_results.append(r); save_results()

# Include Stage D winner as reference (R1 LR=1e-3, DROPOUT=0.25)
sd_ref = best_sd.copy(); sd_ref['run'] = 'SD-best(1e3-D25)'
se_compare = se_results + [sd_ref]
print_stage_table(se_compare, 'E', 'LR_MAX')
best_se    = pick_best(se_compare)
FINAL_LR   = best_se['LR_MAX']
FINAL_DROP = best_se['DROPOUT']
print(f"\n  Best joint: LR_MAX={FINAL_LR:.1e}  DROPOUT={FINAL_DROP}")


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ROUND 2 TUNING COMPLETE — FINAL REPORT")
print("=" * 65)

R1_REF_ACC_ACTIVE    = 0.7686
R1_REF_SHORT_PREC    = 0.602
R1_REF_BALANCED_PREC = 0.691

stage_rows = [
    ('A  Short-Prec',  f"g={BEST_GAMMA} SWF={BEST_SWF}", best_sa,  'balanced_prec'),
    ('B  LR-Stable',   f"LR={BEST_B['LR_MAX']:.1e}",     best_sb,  'acc_active'),
    ('C  Batch',       f"BATCH={BEST_BATCH}",             best_sc,  'acc_active'),
    ('D  Augment',     f"N={BEST_NOISE} M={BEST_MASKP}",  best_sd,  'acc_active'),
    ('E  LR x DROP',   f"LR={FINAL_LR:.1e} D={FINAL_DROP}", best_se, 'acc_active'),
]

print(f"\n  R1 reference: acc_active={R1_REF_ACC_ACTIVE:.4f}  "
      f"short_prec={R1_REF_SHORT_PREC:.3f}  balanced_prec={R1_REF_BALANCED_PREC:.4f}\n")
print(f"  {'Stage':<18}  {'Best value':<26}  {'acc_active':>10}  "
      f"{'short_prec':>10}  {'balanced_prec':>13}  {'delta_acc':>9}")
print(f"  {'─'*90}")

for stage, desc, best, metric in stage_rows:
    da = best['acc_active'] - R1_REF_ACC_ACTIVE
    print(f"  {stage:<18}  {desc:<26}  {best['acc_active']:>10.4f}  "
          f"{best['short_prec']:>10.3f}  {best['balanced_prec']:>13.4f}  {da:>+9.4f}")

FINAL_CFG = {
    **BASELINE_R2,
    **BEST_A,
    **BEST_B,
    **BEST_C,
    **BEST_D,
    'LR_MAX':  FINAL_LR,
    'DROPOUT': FINAL_DROP,
}

print(f"\n{'RECOMMENDED R2 CONFIG':=^65}")
for k, v in FINAL_CFG.items():
    changed = '  <-- changed' if v != BASELINE_R2.get(k) else ''
    print(f"  {k:<26} {str(v):<14}{changed}")

total_min = sum(r['elapsed_min'] for r in all_results)
print(f"\n  Total runs : {len(all_results)}")
print(f"  Total time : {total_min:.0f} min  ({total_min/60:.1f} h)")
print(f"  Results    : {RESULTS_CSV}")
