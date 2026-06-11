"""
btc_lstm_tuning.py
Sequential hyperparameter tuning for BTC LSTM v5.
25 experiments across 6 stages; each stage locks the best value from the
previous before varying the next parameter group.
Results are appended to tuning_results.csv after every run so the job can
be inspected (or resumed manually) at any point.
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
RESULTS_CSV = os.path.join(SAVE_DIR, 'tuning_results.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Estimated total time: ~8–10 hours (25 runs × ~20 min each)\n")


# ─────────────────────────────────────────────
# UTILITIES  (identical to btc_lstm_train.py)
# ─────────────────────────────────────────────
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
    macd  = (ema12 - ema26) / (atr14 + 1e-9)
    macds = macd.ewm(span=9).mean()
    d[f'{px}macd']  = macd
    d[f'{px}macds'] = macds
    d[f'{px}macdh'] = macd - macds

    d[f'{px}atr14'] = atr14 / (c + 1e-9)
    d[f'{px}atr7']  = atr7  / (c + 1e-9)
    d[f'{px}atr_r'] = atr7  / (atr14 + 1e-9)

    bm = c.rolling(20).mean(); bstd = c.rolling(20).std()
    bup = bm + 2*bstd; bdn = bm - 2*bstd
    d[f'{px}bbw'] = (bup - bdn) / (bm + 1e-9)
    d[f'{px}bbp'] = (c - bdn)   / (bup - bdn + 1e-9)

    ll14 = l.rolling(14).min(); hh14 = h.rolling(14).max()
    k = (c - ll14) / (hh14 - ll14 + 1e-9)
    d[f'{px}stk'] = k
    d[f'{px}std'] = k.rolling(3).mean()
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
    d[f'{px}skew20']  = ret1.rolling(20).skew()
    d[f'{px}kurt20']  = ret1.rolling(20).kurt()
    d[f'{px}skew10']  = ret1.rolling(10).skew()
    d[f'{px}autocorr']= ret1.rolling(20).corr(ret1.shift(1))

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
            tp_hit = high_arr[j] >= tp_lvl
            sl_hit = low_arr[j]  <= sl_lvl
            if tp_hit and sl_hit: break
            elif tp_hit: labels[i] = 1; break
            elif sl_hit: labels[i] = 0; break
    return labels


# ─────────────────────────────────────────────
# MODEL  (parameterised — no globals)
# ─────────────────────────────────────────────
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
        self.H   = n_heads; self.dh = d_model // n_heads
        self.scl = self.dh ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop= nn.Dropout(dropout)
        self.rope= RoPEEmbedding(self.dh)
    def forward(self, x):
        B, L, D = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, L, self.H, self.dh).transpose(1, 2)
        K = K.view(B, L, self.H, self.dh).transpose(1, 2)
        V = V.view(B, L, self.H, self.dh).transpose(1, 2)
        cos, sin = self.rope(L, x.device)
        Q, K = _apply_rope(Q, K, cos, sin)
        logits = (Q @ K.transpose(-2, -1)) * self.scl + _alibi_bias(self.H, L, x.device)
        attn   = self.drop(logits.softmax(dim=-1))
        return self.out((attn @ V).transpose(1, 2).reshape(B, L, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n1  = nn.LayerNorm(d_model)
        self.attn= RelativeAttention(d_model, n_heads, dropout)
        self.n2  = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
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
        self.se    = SqueezeExcite(n_feat)
        self.embed = PatchEmbed(n_feat, cfg['PATCH_SIZE'], dm)
        self.blocks= nn.ModuleList([
            TransformerBlock(dm, cfg['N_HEADS'], cfg['D_FF'], cfg['DROPOUT'])
            for _ in range(cfg['N_LAYERS'])
        ])
        self.head  = nn.Sequential(
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


# ─────────────────────────────────────────────
# DATA LOADING  (runs once before the sweep)
# ─────────────────────────────────────────────
def load_csv(path):
    wanted = {'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Number of trades', 'Quote asset volume'}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    df = df.rename(columns={
        'Open time': 'timestamp',
        'Taker buy base asset volume': 'taker_buy_vol',
        'Number of trades': 'n_trades',
        'Quote asset volume': 'quote_vol',
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
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


# ─────────────────────────────────────────────
# BASELINE CONFIG  (current v5)
# ─────────────────────────────────────────────
BASELINE = dict(
    D_MODEL=128, N_HEADS=8, N_LAYERS=4, D_FF=512,
    DROPOUT=0.20, LABEL_SMOOTH=0.10,
    LR_MAX=3e-4, WARMUP_EP=10, PATIENCE=20,
    BATCH=128, EPOCHS=100,
    AUG_NOISE=0.02, AUG_MASK_P=0.10,
    SEQ_LEN=64, PATCH_SIZE=4,
    TP_PCT=0.015, SL_PCT=0.03, TB_TIMEOUT=4,
    LONG_THRESH=0.60, SHORT_THRESH=0.40,
)


# ─────────────────────────────────────────────
# run_experiment
# ─────────────────────────────────────────────
def make_seq(X, y, seq):
    Xs = np.stack([X[i - seq:i] for i in range(seq, len(X) + 1)])
    return Xs, y[seq - 1:]


def run_experiment(cfg, run_label, stage_name):
    t0 = time.time()
    print(f"\n{'─'*65}")
    print(f"[{run_label}]  stage={stage_name}")

    # Labels
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
                          batch_size=BATCH, shuffle=True,  num_workers=0, drop_last=False)
    val_dl   = DataLoader(SeqDS(Xva_s, yva_s), batch_size=BATCH, shuffle=False, num_workers=0)
    test_dl  = DataLoader(SeqDS(Xte_s, yte_s), batch_size=BATCH, shuffle=False, num_workers=0)

    n_feat = Xtr_s.shape[2]
    model  = TemporalTransformer(n_feat, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  train_seq={Xtr_s.shape[0]:,}  "
          f"labels Long={counts[1]:,} Short={counts[0]:,}", flush=True)

    w         = torch.tensor([total/(2*counts[0]), total/(2*counts[1])],
                              dtype=torch.float32).to(device)
    criterion = FocalLS(gamma=2.0, smooth=cfg['LABEL_SMOOTH'], weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LR_MAX'], weight_decay=1e-3)

    def lr_lambda(ep):
        if ep < cfg['WARMUP_EP']: return (ep + 1) / cfg['WARMUP_EP']
        p = (ep - cfg['WARMUP_EP']) / max(1, cfg['EPOCHS'] - cfg['WARMUP_EP'])
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float('inf'); best_state = None; no_imp = 0; ep_stopped = cfg['EPOCHS']

    for ep in range(1, cfg['EPOCHS'] + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

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
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
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

    elapsed = time.time() - t0
    result  = dict(
        run=run_label, stage=stage_name,
        val_loss=round(best_val, 5),
        acc_argmax=round(acc_am, 4),
        acc_active=round(acc_ac, 4),
        n_active=int(mask.sum()),
        pct_filtered=round(100 * (sig == -1).sum() / len(sig), 1),
        long_prec=round(rpt.get('Long',  {}).get('precision', 0), 3),
        long_rec= round(rpt.get('Long',  {}).get('recall',    0), 3),
        short_prec=round(rpt.get('Short',{}).get('precision', 0), 3),
        short_rec= round(rpt.get('Short',{}).get('recall',    0), 3),
        ep_stopped=ep_stopped, n_params=n_params,
        elapsed_min=round(elapsed / 60, 1),
        **cfg,
    )

    print(f"  => val_loss={result['val_loss']}  acc_argmax={result['acc_argmax']}  "
          f"acc_active={result['acc_active']}  n_active={result['n_active']}  "
          f"ep={result['ep_stopped']}  {result['elapsed_min']}min", flush=True)

    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
all_results = []

def save_results():
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

def pick_best(results):
    return max(results, key=lambda r: (r['acc_active'], -r['val_loss']))

def print_stage_table(results, stage_name, varied_key):
    best = pick_best(results)
    print(f"\n{'Stage ' + stage_name + ' summary — varying ' + varied_key:─^65}")
    print(f"  {'run':<14}  {varied_key:<18}  {'val_loss':>9}  "
          f"{'acc_argmax':>10}  {'acc_active':>10}  {'n_active':>8}  {'ep':>4}")
    for r in sorted(results, key=lambda x: x['acc_active'], reverse=True):
        tag = '  <-- best' if r is best else ''
        val = str(r.get(varied_key, '?'))
        print(f"  {r['run']:<14}  {val:<18}  {r['val_loss']:>9.5f}  "
              f"{r['acc_argmax']:>10.4f}  {r['acc_active']:>10.4f}  "
              f"{r['n_active']:>8,}  {r['ep_stopped']:>4}{tag}")


# ═══════════════════════════════════════════════════════════════════
# STAGE 1 — Learning Rate  (5 runs)
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("STAGE 1 — Learning Rate")
print("=" * 65)

s1_results = []
for i, lr in enumerate([1e-4, 2e-4, 3e-4, 5e-4, 1e-3], 1):
    cfg = {**BASELINE, 'LR_MAX': lr}
    r   = run_experiment(cfg, f'S1-R{i}', 'LR')
    all_results.append(r); s1_results.append(r); save_results()

print_stage_table(s1_results, '1', 'LR_MAX')
best_s1  = pick_best(s1_results)
BEST_LR  = best_s1['LR_MAX']
print(f"\n  Locked: LR_MAX = {BEST_LR:.1e}")


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 — Model Capacity  (4 runs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 2 — Model Capacity")
print("=" * 65)

s2_variants = [
    ('Small',  dict(D_MODEL=64,  N_HEADS=4, N_LAYERS=3, D_FF=256)),
    ('Deep',   dict(D_MODEL=128, N_HEADS=8, N_LAYERS=6, D_FF=512)),
    ('Wide',   dict(D_MODEL=192, N_HEADS=8, N_LAYERS=4, D_FF=768)),
    ('Shallow',dict(D_MODEL=256, N_HEADS=8, N_LAYERS=3, D_FF=512)),
]
s2_results = []
for name, cap in s2_variants:
    cfg = {**BASELINE, 'LR_MAX': BEST_LR, **cap}
    r   = run_experiment(cfg, f'S2-{name}', 'Capacity')
    all_results.append(r); s2_results.append(r); save_results()

# include the Stage-1 baseline-capacity run for comparison
s1_ref = next((r for r in s1_results if abs(r['LR_MAX'] - BEST_LR) < 1e-10), None)
s2_compare = s2_results + ([s1_ref] if s1_ref else [])
print_stage_table(s2_compare, '2', 'D_MODEL')
best_s2   = pick_best(s2_compare)
BEST_D    = best_s2['D_MODEL'];   BEST_NLAY = best_s2['N_LAYERS']
BEST_HEAD = best_s2['N_HEADS'];   BEST_DFF  = best_s2['D_FF']
print(f"\n  Locked: D_MODEL={BEST_D}  N_LAYERS={BEST_NLAY}  "
      f"N_HEADS={BEST_HEAD}  D_FF={BEST_DFF}")

BEST_CAP = dict(D_MODEL=BEST_D, N_HEADS=BEST_HEAD, N_LAYERS=BEST_NLAY, D_FF=BEST_DFF)


# ═══════════════════════════════════════════════════════════════════
# STAGE 3 — Sequence Context  (4 runs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 3 — Sequence Context  (SEQ_LEN / PATCH_SIZE)")
print("=" * 65)

s3_variants = [
    ('32p4',  dict(SEQ_LEN=32,  PATCH_SIZE=4)),   # 8 patch tokens
    ('64p2',  dict(SEQ_LEN=64,  PATCH_SIZE=2)),   # 32 patch tokens
    ('96p4',  dict(SEQ_LEN=96,  PATCH_SIZE=4)),   # 24 patch tokens
    ('128p8', dict(SEQ_LEN=128, PATCH_SIZE=8)),   # 16 patch tokens
]
s3_results = []
for name, ctx in s3_variants:
    cfg = {**BASELINE, 'LR_MAX': BEST_LR, **BEST_CAP, **ctx}
    r   = run_experiment(cfg, f'S3-{name}', 'Context')
    all_results.append(r); s3_results.append(r); save_results()

# include best-s2 baseline context for comparison
s2_ref = best_s2
s3_compare = s3_results + [s2_ref]
print_stage_table(s3_compare, '3', 'SEQ_LEN')
best_s3    = pick_best(s3_compare)
BEST_SEQ   = best_s3['SEQ_LEN']
BEST_PATCH = best_s3['PATCH_SIZE']
print(f"\n  Locked: SEQ_LEN={BEST_SEQ}  PATCH_SIZE={BEST_PATCH}")

BEST_CTX = dict(SEQ_LEN=BEST_SEQ, PATCH_SIZE=BEST_PATCH)


# ═══════════════════════════════════════════════════════════════════
# STAGE 4 — Regularisation  (4 runs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 4 — Regularisation  (DROPOUT / LABEL_SMOOTH)")
print("=" * 65)

s4_variants = [
    ('D0.10-L0.10', dict(DROPOUT=0.10, LABEL_SMOOTH=0.10)),
    ('D0.15-L0.10', dict(DROPOUT=0.15, LABEL_SMOOTH=0.10)),
    ('D0.25-L0.10', dict(DROPOUT=0.25, LABEL_SMOOTH=0.10)),
    ('D0.30-L0.15', dict(DROPOUT=0.30, LABEL_SMOOTH=0.15)),
]
s4_results = []
for name, reg in s4_variants:
    cfg = {**BASELINE, 'LR_MAX': BEST_LR, **BEST_CAP, **BEST_CTX, **reg}
    r   = run_experiment(cfg, f'S4-{name}', 'Regularisation')
    all_results.append(r); s4_results.append(r); save_results()

s4_ref     = best_s3
s4_compare = s4_results + [s4_ref]
print_stage_table(s4_compare, '4', 'DROPOUT')
best_s4    = pick_best(s4_compare)
BEST_DROP  = best_s4['DROPOUT']
BEST_LS    = best_s4['LABEL_SMOOTH']
print(f"\n  Locked: DROPOUT={BEST_DROP}  LABEL_SMOOTH={BEST_LS}")

BEST_REG = dict(DROPOUT=BEST_DROP, LABEL_SMOOTH=BEST_LS)


# ═══════════════════════════════════════════════════════════════════
# STAGE 5 — Label Parameters  (5 runs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 5 — Label Parameters  (TP_PCT / SL_PCT / TB_TIMEOUT)")
print("=" * 65)

s5_variants = [
    ('TP1-SL1-T4',   dict(TP_PCT=0.010, SL_PCT=0.010, TB_TIMEOUT=4)),  # tight sym
    ('TP15-SL15-T4', dict(TP_PCT=0.015, SL_PCT=0.015, TB_TIMEOUT=4)),  # medium sym
    ('TP2-SL2-T4',   dict(TP_PCT=0.020, SL_PCT=0.020, TB_TIMEOUT=4)),  # wide sym
    ('TP15-SL3-T4',  dict(TP_PCT=0.015, SL_PCT=0.030, TB_TIMEOUT=4)),  # current (asymmetric)
    ('TP15-SL3-T6',  dict(TP_PCT=0.015, SL_PCT=0.030, TB_TIMEOUT=6)),  # current + wider timeout
]
s5_results = []
for name, lbl in s5_variants:
    cfg = {**BASELINE, 'LR_MAX': BEST_LR, **BEST_CAP, **BEST_CTX, **BEST_REG, **lbl}
    r   = run_experiment(cfg, f'S5-{name}', 'Labels')
    all_results.append(r); s5_results.append(r); save_results()

print_stage_table(s5_results, '5', 'TP_PCT')
best_s5  = pick_best(s5_results)
BEST_TP  = best_s5['TP_PCT']
BEST_SL  = best_s5['SL_PCT']
BEST_TT  = best_s5['TB_TIMEOUT']
print(f"\n  Locked: TP_PCT={BEST_TP}  SL_PCT={BEST_SL}  TB_TIMEOUT={BEST_TT}")


# ═══════════════════════════════════════════════════════════════════
# STAGE 6 — Stability Check  (3 seeds)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 6 — Stability Check  (best config × 3 random seeds)")
print("=" * 65)

FINAL_CFG = {
    **BASELINE,
    'LR_MAX':      BEST_LR,
    **BEST_CAP,
    **BEST_CTX,
    **BEST_REG,
    'TP_PCT':      BEST_TP,
    'SL_PCT':      BEST_SL,
    'TB_TIMEOUT':  BEST_TT,
}
print("\nFinal config to validate:")
for k, v in FINAL_CFG.items():
    changed = '  <-- changed' if v != BASELINE.get(k) else ''
    print(f"  {k:<22} {str(v):<12}{changed}")

s6_results = []
for seed in [42, 123, 7]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    r = run_experiment(FINAL_CFG, f'S6-seed{seed}', 'Stability')
    r['seed'] = seed
    all_results.append(r); s6_results.append(r); save_results()

accs = [r['acc_active'] for r in s6_results]
print(f"\nStability — acc_active:  mean={np.mean(accs):.4f}  "
      f"std={np.std(accs):.4f}  min={min(accs):.4f}  max={max(accs):.4f}")


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TUNING COMPLETE — FINAL REPORT")
print("=" * 65)

stage_rows = [
    ('1  LR',          f"LR_MAX = {BEST_LR:.1e}",                        best_s1),
    ('2  Capacity',    f"D={BEST_D} L={BEST_NLAY} H={BEST_HEAD}",        best_s2),
    ('3  Context',     f"SEQ={BEST_SEQ} PATCH={BEST_PATCH}",             best_s3),
    ('4  Regularise',  f"DROP={BEST_DROP} LS={BEST_LS}",                 best_s4),
    ('5  Labels',      f"TP={BEST_TP} SL={BEST_SL} TT={BEST_TT}",       best_s5),
]

# baseline reference: S1 run with LR=3e-4 (original v5 LR)
baseline_ref = next((r for r in s1_results if abs(r['LR_MAX'] - 3e-4) < 1e-10), s1_results[0])

print(f"\n  {'Stage':<20}  {'Best value':<32}  {'acc_active':>10}  {'val_loss':>9}")
print(f"  {'─'*75}")
print(f"  {'Baseline v5':<20}  {'(original config)':<32}  "
      f"{baseline_ref['acc_active']:>10.4f}  {baseline_ref['val_loss']:>9.5f}")
for stage, desc, best in stage_rows:
    gain = best['acc_active'] - baseline_ref['acc_active']
    print(f"  {stage:<20}  {desc:<32}  {best['acc_active']:>10.4f}  "
          f"{best['val_loss']:>9.5f}  ({gain:+.4f})")

print(f"\n  Stability (final config):  "
      f"mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")

print(f"\n{'RECOMMENDED CONFIG':═^65}")
for k, v in FINAL_CFG.items():
    changed = '  <-- tuned' if v != BASELINE.get(k) else ''
    print(f"  {k:<22} {str(v):<12}{changed}")

total_min = sum(r['elapsed_min'] for r in all_results)
print(f"\n  Total runs : {len(all_results)}")
print(f"  Total time : {total_min:.0f} min  ({total_min/60:.1f} h)")
print(f"  Results CSV: {RESULTS_CSV}")









