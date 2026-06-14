"""
BTC Multi-Timeframe Temporal Transformer v3
Long/Short Signal Prediction (4h target)

Advanced improvements over v2:

  FEATURES
    Taker buy pressure ratio — order-flow microstructure (from raw exchange data)
    Delta volume — (buy - sell) imbalance
    Trade intensity — number of trades vs 20-period average
    Ichimoku Cloud — Tenkan/Kijun diff, price vs cloud, cloud thickness
    CCI (Commodity Channel Index) and Williams %R
    Rolling skewness and kurtosis of returns (distribution shape)
    Realized volatility at 10 and 20 periods
    Return autocorrelation (lag-1) — momentum/mean-reversion detector
    Cross-timeframe RSI divergence (4h vs 1d, 4h vs 1h)
    Cross-timeframe MACD histogram divergence and volume ratio

  NETWORK
    Squeeze-and-Excitation (SE) channel gating — learns which features matter
    Patch embedding — groups 4 timesteps into one token (reduces noise)
    Temporal Transformer encoder (4 layers, 8 heads, pre-LayerNorm)
    Learnable positional encoding
    Mean pooling → 2-layer MLP classifier

  TRAINING
    Focal loss + label smoothing (0.1) — avoids overconfident plateau
    Gaussian jitter + random feature masking (data augmentation)
    LR warmup + cosine decay
"""

import os, sys, warnings
# Force UTF-8 stdout so the unicode bar chars in the attention report
# don't crash on Windows' default GBK console encoding.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR     = r"D:\Document\LLLLLLLLLLLLL\DATA"
SAVE_DIR     = r"D:\Document\LLLLLLLLLLLLL"
SEQ_LEN      = 64
PATCH_SIZE   = 4        # 64 steps → 16 patch tokens
# Triple-Barrier Label parameters (Lopez de Prado)
TP_PCT       = 0.015   # take-profit barrier: 
SL_PCT       = 0.03    # stop-loss barrier:   
TB_TIMEOUT   = 4      # vertical barrier:     
BATCH        = 64
EPOCHS       = 100
LR_MAX       = 1e-3
WARMUP_EP    = 20
PATIENCE     = 20
D_MODEL      = 256
N_HEADS      = 8
N_LAYERS     = 3
D_FF         = 512
DROPOUT      = 0.25
LABEL_SMOOTH = 0.10
AUG_NOISE    = 0.02
AUG_MASK_P   = 0.10
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.85
LONG_THRESH  = 0.60
SHORT_THRESH = 0.40
# Adaptive threshold config — volatility-regime gating
VOL_FEATURE     = '4h_atr14'      # scale-invariant ATR used to classify regime
THRESH_LOW_VOL  = (0.55, 0.45)    # (long_thr, short_thr) quiet regime
THRESH_HIGH_VOL = (0.72, 0.28)    # (long_thr, short_thr) volatile regime
SHORT_WEIGHT_FACTOR = 1.0        # extra multiplier on Short class weight (R2 tuned)


# ─────────────────────────────────────────────
# 1. LOAD DATA  (include microstructure columns)
# ─────────────────────────────────────────────
def load_csv(path):
    wanted = {'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Number of trades',
              'Quote asset volume'}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    df = df.rename(columns={
        'Open time':                  'timestamp',
        'Taker buy base asset volume':'taker_buy_vol',
        'Number of trades':           'n_trades',
        'Quote asset volume':         'quote_vol',
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

print("Loading CSVs …")
df_15m = load_csv(os.path.join(DATA_DIR, 'btc_15m_data_2018_to_2025.csv'))
df_1h  = load_csv(os.path.join(DATA_DIR, 'btc_1h_data_2018_to_2025.csv'))
df_4h  = load_csv(os.path.join(DATA_DIR, 'btc_4h_data_2018_to_2025.csv'))
df_1d  = load_csv(os.path.join(DATA_DIR, 'btc_1d_data_2018_to_2025.csv'))
print(f"  15m:{len(df_15m):,}  1h:{len(df_1h):,}  4h:{len(df_4h):,}  1d:{len(df_1d):,}")


# ─────────────────────────────────────────────
# 2. ADVANCED SCALE-INVARIANT INDICATORS
# ─────────────────────────────────────────────
def atr_ema(h, l, c, period=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def add_indicators(df, px=''):
    d = df.copy()
    c, h, l, v, o = d['Close'], d['High'], d['Low'], d['Volume'], d['Open']

    atr14 = atr_ema(h, l, c, 14)
    atr7  = atr_ema(h, l, c,  7)

    # ── Microstructure (order flow) ──────────────────────────────
    ret1 = c.pct_change()   # used by several microstructure features below

    if 'taker_buy_vol' in d.columns:
        buy_vol  = d['taker_buy_vol']
        sell_vol = v - buy_vol
        buy_r    = buy_vol / (v + 1e-9)                          # ∈ [0,1]
        ofi      = buy_vol - sell_vol                             # signed flow

        # Basic buy-pressure
        d[f'{px}buy_ratio']     = buy_r
        d[f'{px}buy_ratio_ma']  = buy_r.rolling(10).mean()
        d[f'{px}buy_ratio_dev'] = buy_r - buy_r.rolling(20).mean()
        d[f'{px}delta_vol']     = 2 * buy_r - 1                  # ∈ [-1,1]
        d[f'{px}delta_vol_ma']  = d[f'{px}delta_vol'].rolling(10).mean()

        # Cumulative OFI normalised by rolling total volume
        for w in [5, 10, 20]:
            d[f'{px}ofi{w}'] = ofi.rolling(w).sum() / (v.rolling(w).sum() + 1e-9)

        # OFI momentum: short-window OFI minus long-window OFI
        d[f'{px}ofi_mom'] = d[f'{px}ofi5'] - d[f'{px}ofi20']

        # VPIN proxy: |buy − sell| / total over rolling window
        # High VPIN = high toxicity / informed trading activity
        for w in [5, 20]:
            d[f'{px}vpin{w}'] = ofi.abs().rolling(w).sum() / (v.rolling(w).sum() + 1e-9)

        # OFI–price divergence: price going up but sellers dominating → bearish
        price_dir = np.sign(ret1)
        ofi_dir   = np.sign(ofi)
        d[f'{px}ofi_div'] = (price_dir - ofi_dir).rolling(10).mean()  # ∈ [-2,2]

        # Amihud illiquidity: |return| / quote volume traded
        # High = price moves a lot per dollar → low liquidity / high impact
        if 'quote_vol' in d.columns:
            amihud = ret1.abs() / (d['quote_vol'] + 1e-9)
            d[f'{px}amihud'] = amihud / (amihud.rolling(20).mean() + 1e-9)

    if 'n_trades' in d.columns:
        nt = d['n_trades']
        d[f'{px}trade_int'] = nt / (nt.rolling(20).mean() + 1e-9)
        # Avg trade size (volume per trade) — small = retail flow, large = institutional
        avg_trade = v / (nt + 1e-9)
        d[f'{px}trade_size'] = avg_trade / (avg_trade.rolling(20).mean() + 1e-9)

    # ── Volume-synchronized volatility ───────────────────────────
    # Measures how much price moves per unit of volume traded.
    # High = illiquid or high-impact candle; low = smooth/liquid.
    vol_sync = (h - l) / (v + 1e-9)
    d[f'{px}vol_sync'] = vol_sync / (vol_sync.rolling(20).mean() + 1e-9)

    # ── Realized bipower variation (RBV) ─────────────────────────
    # RV  = Σ r²       — includes variance from jumps
    # RBV = Σ|r_t||r_{t-1}| × π/2  — robust estimator that cancels jumps
    # jump_ratio = RV / RBV > 1 signals a jump (news, liquidation cascade)
    rv  = (ret1 ** 2).rolling(20).sum()
    rbv = (ret1.abs() * ret1.shift(1).abs()).rolling(20).sum() * (np.pi / 2)
    d[f'{px}jump_ratio'] = rv / (rbv + 1e-9)      # dimensionless; ~1 = no jump

    # ── Liquidity sweep detection ─────────────────────────────────
    # A sweep: price spikes through a level, triggers stops, then reverses.
    # Signature: long wick + high volume + small net price displacement.
    upper_wick = (h - c.clip(lower=o))  / (atr14 + 1e-9)   # already have body/lower
    lower_wick = (c.clip(upper=o) - l) / (atr14 + 1e-9)
    vsma       = v.rolling(20).mean()
    vr_local   = v / (vsma + 1e-9)
    d[f'{px}upper_sweep'] = upper_wick * vr_local   # wick × volume spike
    d[f'{px}lower_sweep'] = lower_wick * vr_local
    # Range efficiency: 1 = fully directional candle; near 0 = inside-out sweep
    d[f'{px}range_eff']   = ret1.abs() / (((h - l) / c) + 1e-9)

    # ── EMA ratios (scale-invariant) ─────────────────────────────
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

    # ── RSI ──────────────────────────────────────────────────────
    delta = c.diff()
    for per, name in [(7,'rsi7'), (14,'rsi14')]:
        g  = delta.clip(lower=0).ewm(alpha=1/per, min_periods=per).mean()
        ls = (-delta.clip(upper=0)).ewm(alpha=1/per, min_periods=per).mean()
        d[f'{px}{name}'] = (100 - 100/(1 + g/(ls+1e-9))) / 100   # → [0,1]

    # ── MACD (normalised by ATR) ─────────────────────────────────
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    macd  = (ema12 - ema26) / (atr14 + 1e-9)
    macds = macd.ewm(span=9).mean()
    d[f'{px}macd']  = macd
    d[f'{px}macds'] = macds
    d[f'{px}macdh'] = macd - macds

    # ── ATR ──────────────────────────────────────────────────────
    d[f'{px}atr14'] = atr14 / (c + 1e-9)
    d[f'{px}atr7']  = atr7  / (c + 1e-9)
    d[f'{px}atr_r'] = atr7  / (atr14 + 1e-9)

    # ── Bollinger Bands ──────────────────────────────────────────
    bm   = c.rolling(20).mean(); bstd = c.rolling(20).std()
    bup  = bm + 2*bstd; bdn = bm - 2*bstd
    d[f'{px}bbw'] = (bup - bdn) / (bm + 1e-9)
    d[f'{px}bbp'] = (c - bdn) / (bup - bdn + 1e-9)

    # ── Stochastic ───────────────────────────────────────────────
    ll14 = l.rolling(14).min(); hh14 = h.rolling(14).max()
    k    = (c - ll14) / (hh14 - ll14 + 1e-9)
    d[f'{px}stk'] = k
    d[f'{px}std'] = k.rolling(3).mean()

    # ── Williams %R ──────────────────────────────────────────────
    d[f'{px}wpr'] = (hh14 - c) / (hh14 - ll14 + 1e-9)   # 0=overbought, 1=oversold

    # ── CCI ──────────────────────────────────────────────────────
    tp     = (h + l + c) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d[f'{px}cci'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-9) / 200

    # ── Ichimoku Cloud ───────────────────────────────────────────
    tenkan  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun   = (h.rolling(26).max() + l.rolling(26).min()) / 2
    # Current-cloud values (shifted 26 backward = standard Ichimoku convention)
    senA    = ((tenkan + kijun) / 2).shift(26)
    senB    = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    cloud_mid = (senA + senB) / 2
    d[f'{px}ichi_tk']  = (tenkan - kijun) / (c + 1e-9)        # TK cross signal
    d[f'{px}ichi_pos'] = (c - cloud_mid)  / (c + 1e-9)        # price vs cloud centre
    d[f'{px}ichi_cld'] = (senA   - senB)  / (c + 1e-9)        # cloud colour & depth

    # ── Volume ───────────────────────────────────────────────────
    vsma = v.rolling(20).mean()
    d[f'{px}vr']  = v / (vsma + 1e-9)
    d[f'{px}vr5'] = v.rolling(5).mean() / (vsma + 1e-9)
    obv   = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_m = obv.rolling(30).mean(); obv_s = obv.rolling(30).std()
    d[f'{px}obv'] = (obv - obv_m) / (obv_s + 1e-9)

    # ── Returns & higher-order stats ─────────────────────────────
    # ret1 already defined at top of function (used by microstructure)
    for n in [1, 2, 3, 6, 12, 24]:
        d[f'{px}ret{n}'] = c.pct_change(n)
    for n in [10, 20]:
        d[f'{px}rvol{n}']  = ret1.rolling(n).std()          # realized vol
    d[f'{px}skew20']   = ret1.rolling(20).skew()
    d[f'{px}kurt20']   = ret1.rolling(20).kurt()
    d[f'{px}skew10']   = ret1.rolling(10).skew()
    # Lag-1 autocorrelation (positive = momentum, negative = mean-reversion)
    d[f'{px}autocorr'] = ret1.rolling(20).corr(ret1.shift(1))

    # ── Candle structure ─────────────────────────────────────────
    d[f'{px}body']  = (c - o) / (atr14 + 1e-9)
    d[f'{px}hl']    = (h - l) / (c + 1e-9)
    d[f'{px}upper'] = (h - c.clip(lower=o))  / (atr14 + 1e-9)
    d[f'{px}lower'] = (c.clip(upper=o) - l) / (atr14 + 1e-9)

    return d


print("Computing indicators …")
df_15m_f = add_indicators(df_15m, '15m_')
df_1h_f  = add_indicators(df_1h,  '1h_')
df_4h_f  = add_indicators(df_4h,  '4h_')
df_1d_f  = add_indicators(df_1d,  '1d_')


# ─────────────────────────────────────────────
# 3. MERGE ALL TIMEFRAMES → 4H BASE
# ─────────────────────────────────────────────
def ind_cols(df, prefix):
    return ['timestamp'] + [c for c in df.columns if c.startswith(prefix)]

base = df_4h_f.copy()
for other, px in [(df_1h_f,'1h_'), (df_1d_f,'1d_'), (df_15m_f,'15m_')]:
    base = pd.merge_asof(
        base.sort_values('timestamp'),
        other[ind_cols(other, px)].sort_values('timestamp'),
        on='timestamp', direction='backward'
    )
base = base.sort_values('timestamp').reset_index(drop=True)

# ── Cross-timeframe divergence features ─────────────────────────
for a, b, name in [('4h_rsi14','1d_rsi14','div_rsi_4h1d'),
                   ('4h_rsi14','1h_rsi14','div_rsi_4h1h'),
                   ('4h_macdh','1d_macdh','div_macdh_4h1d')]:
    if a in base.columns and b in base.columns:
        base[name] = base[a] - base[b]

if '4h_vr' in base.columns and '1d_vr' in base.columns:
    base['div_vol_4h1d'] = base['4h_vr'] / (base['1d_vr'] + 1e-9)

print(f"Merged shape: {base.shape}")


# ─────────────────────────────────────────────
# 4. LABELS  — Triple-Barrier Method (Lopez de Prado)
# ─────────────────────────────────────────────
# For each candle t, scan forward up to TB_TIMEOUT candles.
# Use each future candle's High/Low so intracandle barrier touches are captured.
#
#   High[t+k] >= TP  →  label 1  (Long:  TP hit first)
#   Low[t+k]  <= SL  →  label 0  (Short: SL hit first)
#   Both in same candle → ambiguous → NaN (dropped)
#   Timeout reached without either → NaN (dropped)
#
# Asymmetric barriers (TP 1.0% > SL 0.7%) encode a positive risk/reward target.

def triple_barrier(close_arr, high_arr, low_arr, tp_pct, sl_pct, timeout):
    n      = len(close_arr)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        ref    = close_arr[i]
        tp_lvl = ref * (1 + tp_pct)
        sl_lvl = ref * (1 - sl_pct)
        for j in range(i + 1, min(i + 1 + timeout, n)):
            tp_hit = high_arr[j] >= tp_lvl
            sl_hit = low_arr[j]  <= sl_lvl
            if tp_hit and sl_hit:
                break               # both barriers in same candle — ambiguous
            elif tp_hit:
                labels[i] = 1
                break
            elif sl_hit:
                labels[i] = 0
                break
        # if loop ends without break → timeout → NaN
    return labels

print("Computing triple-barrier labels …")
base['label'] = triple_barrier(
    base['Close'].values, base['High'].values, base['Low'].values,
    TP_PCT, SL_PCT, TB_TIMEOUT
)
# Report label stats (before dropping timeout rows)
_lbl_valid = base['label'].dropna()
total_tb = len(_lbl_valid)
n_long   = int(_lbl_valid.sum())
n_short  = total_tb - n_long
print(f"After triple-barrier filter: {total_tb:,}  "
      f"Long={n_long:,} ({100*n_long/total_tb:.1f}%)  "
      f"Short={n_short:,} ({100*n_short/total_tb:.1f}%)")
# NOTE: timeout rows (NaN label) are kept in the feature matrix so that
# sequences at inference time use the same full-row context as during training.


# ─────────────────────────────────────────────
# 5. FEATURE MATRIX  (all rows, including timeout)
# ─────────────────────────────────────────────
RAW_COLS = {'timestamp','Open','High','Low','Close','Volume','label',
            'taker_buy_vol','n_trades','quote_vol'}
feat_cols = [c for c in base.columns if c not in RAW_COLS and base[c].nunique() > 1]
base      = base.dropna(subset=feat_cols).reset_index(drop=True)

label_full = base['label'].values          # NaN for timeout rows
X_full     = base[feat_cols].values.astype(np.float32)
ts_full    = base['timestamp'].values
print(f"Features: {len(feat_cols)}   Total rows (incl. timeout): {len(X_full):,}")


# ─────────────────────────────────────────────
# 6. BUILD LABELED SEQUENCE INDEX
# ─────────────────────────────────────────────
# Each entry = (end_idx, label) for a sequence X_full[end_idx-SEQ_LEN : end_idx]
# Only sequences whose last row has a valid (non-NaN) label are kept.
valid_label = ~np.isnan(label_full)
labeled_pos = [(i, int(label_full[i-1]))
               for i in range(SEQ_LEN, len(X_full) + 1)
               if valid_label[i - 1]]

n_labeled = len(labeled_pos)
i_tr = int(n_labeled * TRAIN_FRAC)
i_va = int(n_labeled * VAL_FRAC)

# Scaler is fit on the full rows up to the training boundary (matches btc_predict.py)
train_end_idx = labeled_pos[i_tr - 1][0]   # X_full row index of last training sequence
scaler = RobustScaler(quantile_range=(5, 95))
X_scaled = np.empty_like(X_full)
X_scaled[:train_end_idx] = np.clip(
    scaler.fit_transform(X_full[:train_end_idx]), -6, 6).astype(np.float32)
X_scaled[train_end_idx:] = np.clip(
    scaler.transform(X_full[train_end_idx:]),     -6, 6).astype(np.float32)


# ─────────────────────────────────────────────
# 7. SEQUENCE WINDOWS
# ─────────────────────────────────────────────
def make_seq_from_pos(positions, X_sc, seq):
    Xs = np.stack([X_sc[pos - seq:pos] for pos, _ in positions])
    ys = np.array([lbl for _, lbl in positions], dtype=np.int64)
    return Xs, ys

Xtr_s, ytr_s = make_seq_from_pos(labeled_pos[:i_tr],      X_scaled, SEQ_LEN)
Xva_s, yva_s = make_seq_from_pos(labeled_pos[i_tr:i_va],  X_scaled, SEQ_LEN)
Xte_s, yte_s = make_seq_from_pos(labeled_pos[i_va:],      X_scaled, SEQ_LEN)
ts_te_s = np.array([ts_full[pos - 1] for pos, _ in labeled_pos[i_va:]])

counts = Counter(ytr_s.tolist())
print(f"Split — train:{len(ytr_s):,}  val:{len(yva_s):,}  test:{len(yte_s):,}")
print(f"Sequences — train:{Xtr_s.shape}  val:{Xva_s.shape}  test:{Xte_s.shape}")
print(f"Train — Short:{counts[0]:,}  Long:{counts[1]:,}")


class SeqDS(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            x = x + AUG_NOISE * torch.randn_like(x)              # temporal jitter
            mask = torch.bernoulli(torch.full((x.shape[-1],),     # feature masking
                                              AUG_MASK_P)).bool()
            x[:, mask] = 0.0
        return x, self.y[i]

train_dl = DataLoader(SeqDS(Xtr_s, ytr_s, augment=True),
                      batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0)
val_dl   = DataLoader(SeqDS(Xva_s, yva_s), batch_size=BATCH, shuffle=False, num_workers=0)
test_dl  = DataLoader(SeqDS(Xte_s, yte_s), batch_size=BATCH, shuffle=False, num_workers=0)


# ─────────────────────────────────────────────
# 8. MODEL
# ─────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    """Recalibrate each feature channel by its global temporal importance."""
    def __init__(self, n_feat, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_feat, max(n_feat // r, 16)),
            nn.ReLU(),
            nn.Linear(max(n_feat // r, 16), n_feat),
            nn.Sigmoid(),
        )
    def forward(self, x):       # x: (B, T, F)
        w = self.fc(x.mean(1))  # (B, F) — global avg pool over time
        return x * w.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Fold P consecutive timesteps into one token and project to d_model."""
    def __init__(self, n_feat, patch=PATCH_SIZE, d=D_MODEL):
        super().__init__()
        self.p    = patch
        self.proj = nn.Linear(n_feat * patch, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):               # x: (B, T, F)
        B, T, F = x.shape
        pad = (self.p - T % self.p) % self.p
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = x.reshape(B, -1, self.p * F)
        return self.norm(self.proj(x))   # (B, n_patches, d_model)


# ── Rotary Position Embeddings (RoPE) ───────────────────────────
# Encodes relative distance directly into Q·K dot-products by rotating
# Q and K vectors with position-dependent angles.  No learnable params.
class RoPEEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)           # (L, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)       # (L, dim)
        return emb.cos(), emb.sin()                     # each (L, dim)

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def _apply_rope(q, k, cos, sin):
    cos = cos[None, None]   # (1, 1, L, dh) — broadcast over batch & heads
    sin = sin[None, None]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


# ── ALiBi attention bias ─────────────────────────────────────────
# Adds a linear distance penalty to attention logits so nearby tokens
# are always favoured regardless of learned weights.  No extra params.
def _alibi_slopes(n_heads):
    def slopes(n):
        return [2 ** (-8 * i / n) for i in range(1, n + 1)]
    if (n_heads & (n_heads - 1)) == 0:          # exact power-of-2
        return torch.tensor(slopes(n_heads), dtype=torch.float32)
    p      = 2 ** int(np.floor(np.log2(n_heads)))
    extra  = slopes(2 * p)[0::2][:n_heads - p]
    return torch.tensor(slopes(p) + extra, dtype=torch.float32)

def _alibi_bias(n_heads, seq_len, device):
    slopes = _alibi_slopes(n_heads).to(device)                       # (H,)
    dist   = (torch.arange(seq_len, device=device).unsqueeze(0) -
              torch.arange(seq_len, device=device).unsqueeze(1)
              ).abs().float()                                          # (L, L)
    return -slopes.view(-1, 1, 1) * dist.unsqueeze(0)                # (H, L, L)


# ── Fused RoPE + ALiBi multi-head attention ──────────────────────
class RelativeAttention(nn.Module):
    """
    RoPE  — positions enter via Q/K rotation → captures relative offset.
    ALiBi — linear distance penalty on logits → explicit recency bias.
    Both are parameter-free positional methods; they complement each other:
    RoPE handles the 'where', ALiBi handles the 'how far matters'.
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.H    = n_heads
        self.dh   = d_model // n_heads
        self.scl  = self.dh ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model,     bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RoPEEmbedding(self.dh)

    def forward(self, x, return_attn=False):            # x: (B, L, D)
        B, L, D = x.shape
        H, dh   = self.H, self.dh

        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, L, H, dh).transpose(1, 2)    # (B, H, L, dh)
        K = K.view(B, L, H, dh).transpose(1, 2)
        V = V.view(B, L, H, dh).transpose(1, 2)

        cos, sin = self.rope(L, x.device)           # each (L, dh)
        Q, K     = _apply_rope(Q, K, cos, sin)

        logits = (Q @ K.transpose(-2, -1)) * self.scl      # (B, H, L, L)
        logits = logits + _alibi_bias(H, L, x.device)      # + ALiBi recency bias
        attn_w = logits.softmax(dim=-1)                     # save pre-dropout for analysis
        attn   = self.drop(attn_w)

        out = (attn @ V).transpose(1, 2).reshape(B, L, D)
        out = self.out(out)
        if return_attn:
            return out, attn_w.detach()
        return out


# ── Pre-norm Transformer block ────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1   = nn.LayerNorm(D_MODEL)
        self.attn = RelativeAttention()
        self.n2   = nn.LayerNorm(D_MODEL)
        self.ff   = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_w = self.attn(self.n1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ff(self.n2(x))
            return x, attn_w
        x = x + self.attn(self.n1(x))   # pre-norm + residual
        x = x + self.ff(self.n2(x))
        return x


# ── Full model  (SE → PatchEmbed → Transformer) ──────────────────
class TemporalTransformer(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.se     = SqueezeExcite(n_feat)
        self.embed  = PatchEmbed(n_feat)
        # RoPE + ALiBi handle all positional information — no learned pos embedding
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.head   = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 2),
        )

    def forward(self, x, return_attn=False):            # x: (B, T, F)
        x   = self.se(x)
        tok = self.embed(x)             # (B, N_patches, D_MODEL)
        if return_attn:
            layer_attns = []
            for blk in self.blocks:
                tok, aw = blk(tok, return_attn=True)
                layer_attns.append(aw)             # each (B, H, Np, Np)
            return self.head(tok.mean(1)), layer_attns
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(tok.mean(1))   # mean pool → classifier


# ─────────────────────────────────────────────
# 9. FOCAL LOSS + LABEL SMOOTHING
# ─────────────────────────────────────────────
class FocalLS(nn.Module):
    def __init__(self, gamma=2.0, smooth=LABEL_SMOOTH, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.smooth = smooth
        self.weight = weight

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

n_feat = Xtr_s.shape[2]
model  = TemporalTransformer(n_feat).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

total = sum(counts.values())
w = torch.tensor([total/(2*counts[0]) * SHORT_WEIGHT_FACTOR, total/(2*counts[1])],
                 dtype=torch.float32).to(device)
criterion = FocalLS(gamma=2.0, smooth=LABEL_SMOOTH, weight=w)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-3)

def lr_lambda(ep):
    if ep < WARMUP_EP:
        return (ep + 1) / WARMUP_EP
    p = (ep - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return 0.5 * (1 + np.cos(np.pi * p))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# 10. TRAINING LOOP
# ─────────────────────────────────────────────
best_val_loss = float('inf')
best_state    = None
no_improve    = 0
hist          = {'tl':[], 'vl':[], 'va':[]}

print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}")
print("-"*48)

for ep in range(1, EPOCHS+1):
    model.train()
    tl = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += criterion(model(xb), yb).item()   # recompute for logging (avoids extra storage)
    tl /= len(train_dl)
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
    va  = accuracy_score(vtrue, vpreds)

    hist['tl'].append(tl); hist['vl'].append(vl); hist['va'].append(va)
    cur_lr = optimizer.param_groups[0]['lr']

    if vl < best_val_loss:
        best_val_loss = vl
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve    = 0
        marker = ' *'          # mark best epoch
    else:
        no_improve += 1
        marker = f'  (no improve {no_improve}/{PATIENCE})'

    print(f"{ep:4d}  {tl:8.4f}  {vl:8.4f}  {va:7.4f}  {cur_lr:9.2e}{marker}")

    if no_improve >= PATIENCE:
        print(f"  → Early stop at epoch {ep}")
        break

print(f"\nBest val loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────
# 11. TEST EVALUATION
# ─────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()

tpreds, tprobs, ttrue = [], [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb   = xb.to(device)
        out  = model(xb)
        prbs = torch.softmax(out, 1).cpu().numpy()
        tpreds.extend(out.argmax(1).cpu().tolist())
        tprobs.extend(prbs.tolist())
        ttrue.extend(yb.tolist())

tpreds = np.array(tpreds); tprobs = np.array(tprobs); ttrue = np.array(ttrue)
p_long = tprobs[:, 1]

# ── Adaptive thresholds (volatility-regime gating) ───────────────
# Use the ATR value at the last timestep of each test sequence to
# classify the current regime, then tighten/loosen confidence gates.
vol_idx = feat_cols.index(VOL_FEATURE) if VOL_FEATURE in feat_cols else None
p33 = p67 = None   # populated below; saved in checkpoint so btc_predict.py uses the same split
if vol_idx is not None:
    vol_tr   = Xtr_s[:, -1, vol_idx]                        # training ATR (scaled)
    p33, p67 = np.percentile(vol_tr, [33, 67])              # regime boundaries
    vol_te   = Xte_s[:len(tpreds), -1, vol_idx]             # test ATR (scaled)
    long_thr  = np.where(vol_te > p67, THRESH_HIGH_VOL[0],
                np.where(vol_te < p33, THRESH_LOW_VOL[0],  LONG_THRESH))
    short_thr = np.where(vol_te > p67, THRESH_HIGH_VOL[1],
                np.where(vol_te < p33, THRESH_LOW_VOL[1],  SHORT_THRESH))
    regime    = np.where(vol_te > p67, 'high',
                np.where(vol_te < p33, 'low', 'mid'))
    n_low  = (vol_te < p33).sum()
    n_mid  = ((vol_te >= p33) & (vol_te <= p67)).sum()
    n_high = (vol_te > p67).sum()
    print(f"\nVol-regime split  low:{n_low:,} ({THRESH_LOW_VOL[0]}/{THRESH_LOW_VOL[1]})  "
          f"mid:{n_mid:,} ({LONG_THRESH}/{SHORT_THRESH})  "
          f"high:{n_high:,} ({THRESH_HIGH_VOL[0]}/{THRESH_HIGH_VOL[1]})")
else:
    long_thr  = np.full(len(tpreds), LONG_THRESH)
    short_thr = np.full(len(tpreds), SHORT_THRESH)
    regime    = np.full(len(tpreds), 'mid')

sig = np.where(p_long >= long_thr,  1,
     np.where(p_long <= short_thr, 0, -1))
mask = sig != -1

print("\n" + "="*55)
print("TEST SET (argmax baseline)")
print("="*55)
print(f"Accuracy: {accuracy_score(ttrue, tpreds):.4f}")
print(classification_report(ttrue, tpreds, target_names=['Short','Long']))

print("\n" + "="*55)
print("ADAPTIVE CONFIDENCE-FILTERED SIGNALS")
print("="*55)
print(f"  LONG   : {(sig==1).sum():,}")
print(f"  SHORT  : {(sig==0).sum():,}")
print(f"  NEUTRAL: {(sig==-1).sum():,}  ({100*(sig==-1).sum()/len(sig):.1f}% filtered)")
if mask.sum() > 0:
    print(f"\nAccuracy on active signals: {accuracy_score(ttrue[mask], sig[mask]):.4f}")
    print(classification_report(ttrue[mask], sig[mask], target_names=['Short','Long']))
    print("Confusion matrix:")
    print(confusion_matrix(ttrue[mask], sig[mask]))
    # ── Per-regime breakdown ─────────────────────────────────────
    print("\nPer-regime accuracy:")
    for reg, lt, st in [('low',  THRESH_LOW_VOL[0],  THRESH_LOW_VOL[1]),
                        ('mid',  LONG_THRESH,         SHORT_THRESH),
                        ('high', THRESH_HIGH_VOL[0],  THRESH_HIGH_VOL[1])]:
        rm = (regime == reg) & mask
        if rm.sum() > 0:
            acc = accuracy_score(ttrue[rm], sig[rm])
            print(f"  {reg:4s}  thr={lt}/{st}  "
                  f"n={rm.sum():4,}  acc={acc:.4f}")

print(f"\nProb(Long) — mean:{p_long.mean():.3f}  std:{p_long.std():.3f}  "
      f"min:{p_long.min():.3f}  max:{p_long.max():.3f}")


# ─────────────────────────────────────────────
# 12. SAVE
# ─────────────────────────────────────────────
ckpt = os.path.join(SAVE_DIR, 'btc_lstm_model.pth')
torch.save({
    'model_state': best_state,
    'model_cfg':   dict(n_feat=n_feat, d_model=D_MODEL, n_heads=N_HEADS,
                        n_layers=N_LAYERS, d_ff=D_FF, drop=DROPOUT,
                        patch_size=PATCH_SIZE),
    'scaler':      scaler,
    'feat_cols':   feat_cols,
    'seq_len':     SEQ_LEN,
    'history':     hist,
    'vol_p33':     float(p33) if p33 is not None else None,
    'vol_p67':     float(p67) if p67 is not None else None,
}, ckpt)
print(f"\nCheckpoint → {ckpt}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(hist['tl'], label='Train'); axes[0].plot(hist['vl'], label='Val')
axes[0].set_title('Focal + Label-Smooth Loss'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(hist['va'], color='green')
axes[1].axhline(0.5, color='gray', linestyle='--', label='baseline')
axes[1].set_title('Val Accuracy'); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
curves = os.path.join(SAVE_DIR, 'training_curves.png')
plt.savefig(curves, dpi=150); plt.close()
print(f"Curves → {curves}")

sig_df = pd.DataFrame({
    'timestamp':  ts_te_s[:len(tpreds)],
    'true_label': ttrue,
    'prob_short': tprobs[:, 0],
    'prob_long':  p_long,
    'vol_regime': regime,
    'long_thr':   long_thr,
    'short_thr':  short_thr,
    'signal':     np.where(sig == 1, 'LONG', np.where(sig == 0, 'SHORT', 'NEUTRAL')),
})
sig_df.to_csv(os.path.join(SAVE_DIR, 'btc_signals.csv'), index=False)
active = sig_df[sig_df['signal'] != 'NEUTRAL']
active.to_csv(os.path.join(SAVE_DIR, 'btc_signals_active.csv'), index=False)
print(f"Signals → btc_signals.csv  ({len(active):,} active)")
print("\nLast 10 active signals:")
print(active.tail(10).to_string(index=False))


# ─────────────────────────────────────────────
# 13. ATTENTION MAP ANALYSIS
# ─────────────────────────────────────────────
# attn[n,l,h,i,j] = query-patch i attends to key-patch j  (softmax over j → rows sum to 1)
# "received attention" for patch j = sum_i attn[n,l,h,i,j]  (column sum)
# Under uniform attention each patch receives N_PATCHES * (1/N_PATCHES) = 1.0
# Values > 1.0 mean the patch is over-attended; < 1.0 means under-attended.

print("\n" + "="*55)
print("ATTENTION MAP ANALYSIS")
print("="*55)
model.eval()
N_PATCHES = SEQ_LEN // PATCH_SIZE   # 16 patch tokens

attn_store = []
ptr = 0
with torch.no_grad():
    for xb, _ in test_dl:
        n = xb.shape[0]
        _, layer_attns = model(xb.to(device), return_attn=True)
        # stack layers → (B, N_layers, H, Np, Np)
        attn_store.append(torch.stack(layer_attns, dim=1).cpu().numpy())
        ptr += n

attn_all = np.concatenate(attn_store, axis=0)   # (N_test, Nl, H, Np, Np)

# Column-sum over queries → received attention, then avg over layers & heads
recv = attn_all.sum(axis=-2).mean(axis=(1, 2))   # (N_test, N_patches)

patch_pos = np.arange(N_PATCHES)

# ── 1. Recency profile ────────────────────────────────────────────
print(f"\nReceived attention per patch  (uniform baseline = 1.000):")
print(f"  {'p':>2}  [steps]  recv    bar")
for p in range(N_PATCHES):
    val = recv[:, p].mean()
    bar = '█' * int(val * 16)
    tag = '  ← most recent' if p == N_PATCHES - 1 else ''
    print(f"  {p:2d}  [{p*PATCH_SIZE:2d}–{p*PATCH_SIZE+PATCH_SIZE-1:2d}]  "
          f"{val:.3f}  {bar}{tag}")

early = recv[:, :4].mean()
late  = recv[:, -4:].mean()
print(f"\nEarly patches [0–3]   mean received: {early:.4f}")
print(f"Late  patches [12–15] mean received: {late:.4f}")
print(f"Recency ratio (late / early)        : {late / early:.2f}×")

# ── 2. Long vs Short attention profile ───────────────────────────
long_m  = sig == 1
short_m = sig == 0
if long_m.sum() and short_m.sum():
    print(f"\nRecent-patch attention (last 4 patches):")
    print(f"  Long  signals (n={long_m.sum():,}): {recv[long_m,  -4:].mean():.4f}")
    print(f"  Short signals (n={short_m.sum():,}): {recv[short_m, -4:].mean():.4f}")
    print(f"  Early-patch attention (first 4 patches):")
    print(f"  Long  signals: {recv[long_m,  :4].mean():.4f}")
    print(f"  Short signals: {recv[short_m, :4].mean():.4f}")
    if recv[short_m, :4].mean() > recv[long_m, :4].mean():
        print("  → Short signals draw relatively more attention from older patches")

# ── 3. Per-layer entropy ─────────────────────────────────────────
print(f"\nPer-layer attention entropy  (lower = more focused):")
for l in range(N_LAYERS):
    # average attention matrix for this layer: (Np, Np)
    a   = attn_all[:, l].mean(axis=(0, 1))
    ent = -(a * np.log(a + 1e-9)).sum(axis=-1).mean()
    bar = '░' * int(ent * 8)
    print(f"  Layer {l+1}: {ent:.3f}  {bar}")

# ── 4. OFI divergence correlation ────────────────────────────────
ofi_div_idx = feat_cols.index('4h_ofi_div') if '4h_ofi_div' in feat_cols else None
if ofi_div_idx is not None:
    ofi_vals    = Xte_s[:len(sig), -1, ofi_div_idx]
    recent_attn = recv[:, -4:].mean(axis=1)
    print(f"\nOFI-divergence vs recent-patch attention:")
    for lbl, m in [('Long', long_m), ('Short', short_m)]:
        if m.sum() > 5:
            r = np.corrcoef(np.abs(ofi_vals[m]), recent_attn[m])[0, 1]
            direction = 'positive' if r > 0 else 'negative'
            print(f"  {lbl:5s} (n={m.sum():,}): r = {r:+.3f}  "
                  f"({'|OFI| up → more recent attn' if r > 0 else '|OFI| up → less recent attn'})")

# ── 5. Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A — received attention per patch, grouped by signal type
ax = axes[0, 0]
groups = [
    ('All test',       recv,               'gray',      '-'),
    ('Long  (active)', recv[long_m],       'steelblue', '-'),
    ('Short (active)', recv[short_m],      'tomato',    '-'),
    ('Correct',        recv[(sig != -1) & (sig == ttrue)], 'green',  '--'),
    ('Wrong',          recv[(sig != -1) & (sig != ttrue)], 'orange', '--'),
]
for lbl, data, col, ls in groups:
    if len(data):
        ax.plot(patch_pos, data.mean(0), marker='o', markersize=4,
                label=f'{lbl} (n={len(data)})', color=col, linestyle=ls)
ax.axvspan(11.5, 15.5, alpha=0.08, color='green')
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, label='uniform baseline')
ax.set_xlabel('Patch index  (0=oldest  ·  15=most recent)')
ax.set_ylabel('Mean received attention')
ax.set_title('Received attention per patch — by signal type')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel B — final-layer attention heatmap (all heads averaged)
ax = axes[0, 1]
heat = attn_all[:, -1].mean(axis=(0, 1))   # (Np, Np)
im   = ax.imshow(heat, cmap='plasma', aspect='auto', origin='upper')
ax.set_xlabel('Key patch (attended to)'); ax.set_ylabel('Query patch (attending)')
ax.set_title(f'Layer {N_LAYERS} mean attention map  (all heads, all test samples)')
plt.colorbar(im, ax=ax)

# Panel C — per-layer received attention profile
ax = axes[1, 0]
for l in range(N_LAYERS):
    lr = attn_all[:, l].sum(axis=-2).mean(axis=(0, 1))   # (Np,)
    ax.plot(patch_pos, lr, marker='o', markersize=3, label=f'Layer {l+1}')
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Patch index'); ax.set_ylabel('Mean received attention')
ax.set_title('Received attention by layer  (does focus sharpen with depth?)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel D — correct vs wrong by direction
ax = axes[1, 1]
outcomes = [
    ('Long  correct', (sig==1) & (sig==ttrue), 'steelblue', 'o'),
    ('Long  wrong',   (sig==1) & (sig!=ttrue), 'lightblue', 's'),
    ('Short correct', (sig==0) & (sig==ttrue), 'tomato',    'o'),
    ('Short wrong',   (sig==0) & (sig!=ttrue), 'salmon',    's'),
]
for lbl, m, col, mk in outcomes:
    if m.sum() > 0:
        ax.plot(patch_pos, recv[m].mean(0), marker=mk, markersize=4,
                label=f'{lbl}  n={m.sum()}', color=col)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Patch index'); ax.set_ylabel('Mean received attention')
ax.set_title('Attention pattern — correct vs wrong predictions')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('BTC Transformer  v5 — Attention Map Analysis', fontsize=12, fontweight='bold')
plt.tight_layout()
attn_path = os.path.join(SAVE_DIR, 'attention_analysis.png')
plt.savefig(attn_path, dpi=150); plt.close()
print(f"\nAttention maps → {attn_path}")
