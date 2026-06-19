"""
btc_eth_sol_pooled_train.py — Pooled multi-asset transformer (BTC+ETH+SOL),
validate / test on BTC, with an OPTIONAL learnable asset embedding.

Pushes the data lever further than btc_eth_pooled_train.py:
  * adds SOL training rows (more, more-diverse, higher-vol data)
  * optional asset embedding: a learnable per-asset bias added to the pooled
    representation, so shared layers learn universal patterns while the embedding
    captures per-asset idiosyncrasy (the model applies "BTC-mode" at inference).

Toggle the embedding with the env var:   ASSET_EMB=0  (off)  |  ASSET_EMB=1 (on)

Leakage discipline (identical to the BTC+ETH version):
  * T_train / T_val from BTC's labeled timeline at the same 70/85 fractions
    -> identical BTC test candles as every prior experiment.
  * Train = {BTC,ETH,SOL}[end_ts < T_train].  Val/Test = BTC only.
  * All other-asset data from the val/test era is dropped.

Outputs: btc_eth_sol_pooled_model{_emb}.pth · *_curves.png · *_signals.csv
"""
import os, sys, warnings
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

DATA_DIR     = r"D:\Document\LLLLLLLLLLLLL\DATA"
SAVE_DIR     = r"D:\Document\LLLLLLLLLLLLL"
SEQ_LEN      = 64
PATCH_SIZE   = 4
LONG_TP, LONG_SL, SHORT_TP, SHORT_SL, TB_TIMEOUT = 0.03, 0.015, 0.03, 0.015, 4
BATCH=64; EPOCHS=100; LR_MAX=1e-3; WARMUP_EP=20; PATIENCE=20
D_MODEL=256; N_HEADS=8; N_LAYERS=3; D_FF=512; DROPOUT=0.25
LABEL_SMOOTH=0.10; AUG_NOISE=0.02; AUG_MASK_P=0.10
TRAIN_FRAC=0.70; VAL_FRAC=0.85
LONG_THRESH=0.60; SHORT_THRESH=0.40
VOL_FEATURE='4h_atr14'; THRESH_LOW_VOL=(0.55,0.45); THRESH_HIGH_VOL=(0.72,0.28)
SHORT_WEIGHT_FACTOR=1.0
BTC_ONLY      = os.environ.get('BTC_ONLY','0') == '1'  # train on BTC rows only (baseline)
USE_ASSET_EMB = (os.environ.get('ASSET_EMB','1') == '1') and not BTC_ONLY
USE_CROSS     = os.environ.get('CROSS','1') == '1'   # Approach C: crypto cross-asset features
USE_NEWFEAT   = os.environ.get('NEWFEAT','0') == '1' # Anchored-VWAP + Absorption-Ratio
TAG = ('btconly' if BTC_ONLY else ('emb' if USE_ASSET_EMB else 'noemb')) + \
      ('_cross' if USE_CROSS else '') + ('_nf' if USE_NEWFEAT else '')
ABS_WINDOW = 60   # rolling window for the PCA absorption ratio

ASSETS = {
    'btc': {'15m':'btc_15m_data_2018_to_2025.csv','1h':'btc_1h_data_2018_to_2025.csv',
            '4h':'btc_4h_data_2018_to_2025.csv','1d':'btc_1d_data_2018_to_2025.csv'},
    'eth': {'15m':'eth_15m_data_2018_to_2025.csv','1h':'eth_1h_data_2018_to_2025.csv',
            '4h':'eth_4h_data_2018_to_2025.csv','1d':'eth_1d_data_2018_to_2025.csv'},
    'sol': {'15m':'sol_15m_data_2018_to_2025.csv','1h':'sol_1h_data_2018_to_2025.csv',
            '4h':'sol_4h_data_2018_to_2025.csv','1d':'sol_1d_data_2018_to_2025.csv'},
}
ASSET_ID = {'btc':0, 'eth':1, 'sol':2}
N_ASSETS = len(ASSET_ID)


# ── DATA PIPELINE (verbatim) ───────────────────────────────────────
def load_csv(path):
    wanted = {'Open time','Open','High','Low','Close','Volume',
              'Taker buy base asset volume','Number of trades','Quote asset volume'}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    df = df.rename(columns={'Open time':'timestamp','Taker buy base asset volume':'taker_buy_vol',
                            'Number of trades':'n_trades','Quote asset volume':'quote_vol'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col != 'timestamp': df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

def atr_ema(h, l, c, period=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def add_indicators(df, px=''):
    d = df.copy()
    c, h, l, v, o = d['Close'], d['High'], d['Low'], d['Volume'], d['Open']
    atr14 = atr_ema(h,l,c,14); atr7 = atr_ema(h,l,c,7); ret1 = c.pct_change()
    if 'taker_buy_vol' in d.columns:
        buy_vol = d['taker_buy_vol']; sell_vol = v - buy_vol
        buy_r = buy_vol/(v+1e-9); ofi = buy_vol - sell_vol
        d[f'{px}buy_ratio']=buy_r; d[f'{px}buy_ratio_ma']=buy_r.rolling(10).mean()
        d[f'{px}buy_ratio_dev']=buy_r-buy_r.rolling(20).mean()
        d[f'{px}delta_vol']=2*buy_r-1; d[f'{px}delta_vol_ma']=d[f'{px}delta_vol'].rolling(10).mean()
        for w in [5,10,20]: d[f'{px}ofi{w}']=ofi.rolling(w).sum()/(v.rolling(w).sum()+1e-9)
        d[f'{px}ofi_mom']=d[f'{px}ofi5']-d[f'{px}ofi20']
        for w in [5,20]: d[f'{px}vpin{w}']=ofi.abs().rolling(w).sum()/(v.rolling(w).sum()+1e-9)
        d[f'{px}ofi_div']=(np.sign(ret1)-np.sign(ofi)).rolling(10).mean()
        if 'quote_vol' in d.columns:
            amihud=ret1.abs()/(d['quote_vol']+1e-9); d[f'{px}amihud']=amihud/(amihud.rolling(20).mean()+1e-9)
    if 'n_trades' in d.columns:
        nt=d['n_trades']; d[f'{px}trade_int']=nt/(nt.rolling(20).mean()+1e-9)
        avg_t=v/(nt+1e-9); d[f'{px}trade_size']=avg_t/(avg_t.rolling(20).mean()+1e-9)
    vol_sync=(h-l)/(v+1e-9); d[f'{px}vol_sync']=vol_sync/(vol_sync.rolling(20).mean()+1e-9)
    rv=(ret1**2).rolling(20).sum(); rbv=(ret1.abs()*ret1.shift(1).abs()).rolling(20).sum()*(np.pi/2)
    d[f'{px}jump_ratio']=rv/(rbv+1e-9)
    vsma=v.rolling(20).mean(); vr=v/(vsma+1e-9)
    uw=(h-c.clip(lower=o))/(atr14+1e-9); lw=(c.clip(upper=o)-l)/(atr14+1e-9)
    d[f'{px}upper_sweep']=uw*vr; d[f'{px}lower_sweep']=lw*vr
    d[f'{px}range_eff']=ret1.abs()/((h-l)/c+1e-9)
    ema9=c.ewm(span=9,min_periods=9).mean(); ema21=c.ewm(span=21,min_periods=21).mean()
    ema50=c.ewm(span=50,min_periods=50).mean(); ema200=c.ewm(span=200,min_periods=200).mean()
    d[f'{px}pr9']=c/ema9-1; d[f'{px}pr21']=c/ema21-1; d[f'{px}pr50']=c/ema50-1; d[f'{px}pr200']=c/ema200-1
    d[f'{px}ema9_21']=ema9/ema21-1; d[f'{px}ema21_50']=ema21/ema50-1; d[f'{px}ema50_200']=ema50/ema200-1
    delta=c.diff()
    for per,name in [(7,'rsi7'),(14,'rsi14')]:
        g=delta.clip(lower=0).ewm(alpha=1/per,min_periods=per).mean(); ls=(-delta.clip(upper=0)).ewm(alpha=1/per,min_periods=per).mean()
        d[f'{px}{name}']=(100-100/(1+g/(ls+1e-9)))/100
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
    # ── Anchored-VWAP distance (z-score vs institutional cost basis) ──
    if USE_NEWFEAT and 'timestamp' in d.columns:
        ts = pd.to_datetime(d['timestamp']); iso = ts.dt.isocalendar()
        wk = (iso.year*100 + iso.week).values; mo = (ts.dt.year*100 + ts.dt.month).values
        pv = ((h+l+c)/3*v)
        gw = pd.DataFrame({'pv':pv.values,'v':v.values,'k':wk})
        avwap_w = (gw.groupby('k')['pv'].cumsum()/(gw.groupby('k')['v'].cumsum()+1e-9)).values
        gm = pd.DataFrame({'pv':pv.values,'v':v.values,'k':mo})
        avwap_m = (gm.groupby('k')['pv'].cumsum()/(gm.groupby('k')['v'].cumsum()+1e-9)).values
        d[f'{px}vwap_dist_w'] = (c.values - avwap_w)/(atr14.values+1e-9)   # weekly anchor
        d[f'{px}vwap_dist_m'] = (c.values - avwap_m)/(atr14.values+1e-9)   # monthly anchor
    return d

def triple_barrier(c, h, l, ltp, lsl, stp, ssl, T):
    n=len(c); lab=np.full(n, np.nan)
    for i in range(n-1):
        ref=c[i]; ltl=ref*(1+ltp); lsll=ref*(1-lsl); stl=ref*(1-stp); ssll=ref*(1+ssl)
        lw=sw=None
        for j in range(i+1, min(i+1+T, n)):
            hi,lo=h[j],l[j]
            if lw is None:
                u,dn=hi>=ltl, lo<=lsll
                if u and dn: lw=False
                elif u: lw=True
                elif dn: lw=False
            if sw is None:
                dn,u=lo<=stl, hi>=ssll
                if dn and u: sw=False
                elif dn: sw=True
                elif u: sw=False
            if lw is not None and sw is not None: break
        L,S=(lw is True),(sw is True)
        if L and not S: lab[i]=1
        elif S and not L: lab[i]=0
    return lab

def ind_cols(df, prefix): return ['timestamp']+[c for c in df.columns if c.startswith(prefix)]

def build_base(files):
    d15 = add_indicators(load_csv(os.path.join(DATA_DIR, files['15m'])), '15m_')
    d1h = add_indicators(load_csv(os.path.join(DATA_DIR, files['1h'])),  '1h_')
    d4h = add_indicators(load_csv(os.path.join(DATA_DIR, files['4h'])),  '4h_')
    d1d = add_indicators(load_csv(os.path.join(DATA_DIR, files['1d'])),  '1d_')
    base = d4h.copy()
    for other, px in [(d1h,'1h_'),(d1d,'1d_'),(d15,'15m_')]:
        base = pd.merge_asof(base.sort_values('timestamp'), other[ind_cols(other,px)].sort_values('timestamp'),
                             on='timestamp', direction='backward')
    base = base.sort_values('timestamp').reset_index(drop=True)
    for a,b,name in [('4h_rsi14','1d_rsi14','div_rsi_4h1d'),('4h_rsi14','1h_rsi14','div_rsi_4h1h'),
                     ('4h_macdh','1d_macdh','div_macdh_4h1d')]:
        if a in base.columns and b in base.columns: base[name]=base[a]-base[b]
    if '4h_vr' in base.columns and '1d_vr' in base.columns: base['div_vol_4h1d']=base['4h_vr']/(base['1d_vr']+1e-9)
    base['label'] = triple_barrier(base['Close'].values, base['High'].values, base['Low'].values,
                                   LONG_TP, LONG_SL, SHORT_TP, SHORT_SL, TB_TIMEOUT)
    return base


# ── Approach C: SYMMETRIC cross-asset features (each asset vs the REST) ─────
# Computed on the 4h grid, aligned on timestamp. Leave-one-out market reference
# (mean over the OTHER assets) so the columns mean "this asset vs the rest" for
# every asset → pooling stays valid. All inputs are contemporaneous (<= t) or
# past-windowed, so no lookahead.
X_PREFIX = 'x_'
def rolling_absorption(R, window):
    """Absorption ratio = variance share of the 1st PC over a rolling window.
    R: (T, A) returns matrix (NaN for assets absent at that time). Causal."""
    T = R.shape[0]; out = np.full(T, np.nan)
    for t in range(window, T):
        win = R[t-window:t]; m = ~np.isnan(win).any(axis=0); w = win[:, m]
        if w.shape[1] < 2: continue
        ev = np.linalg.eigvalsh(np.cov(w, rowvar=False)); ev = ev[ev > 0]
        if ev.size: out[t] = ev.max() / ev.sum()
    return out

def add_cross_asset(bases):
    keys = list(bases.keys())
    # per-asset aligned series of returns / vol / momentum on the timestamp index
    comp = {}
    for a in keys:
        b = bases[a].set_index('timestamp')
        c = b['Close']
        comp[a] = pd.DataFrame({
            'r1': c.pct_change(1), 'r6': c.pct_change(6), 'r24': c.pct_change(24),
            'atr': b['4h_atr14'], 'rsi': b['4h_rsi14'],
        })
    idx = sorted(set().union(*[comp[a].index for a in keys]))
    for a in keys: comp[a] = comp[a].reindex(idx)
    # cross-sectional rank of 6-bar return (pct, handles missing assets)
    rank6 = pd.concat([comp[a]['r6'] for a in keys], axis=1, keys=keys).rank(axis=1, pct=True)
    for a in keys:
        others = [k for k in keys if k != a]
        mr1  = pd.concat([comp[k]['r1']  for k in others], axis=1).mean(axis=1)
        mr6  = pd.concat([comp[k]['r6']  for k in others], axis=1).mean(axis=1)
        mr24 = pd.concat([comp[k]['r24'] for k in others], axis=1).mean(axis=1)
        matr = pd.concat([comp[k]['atr'] for k in others], axis=1).mean(axis=1)
        mrsi = pd.concat([comp[k]['rsi'] for k in others], axis=1).mean(axis=1)
        d = comp[a]
        x = pd.DataFrame(index=idx)
        x[f'{X_PREFIX}relret1']  = d['r1']  - mr1          # relative strength (short)
        x[f'{X_PREFIX}relret6']  = d['r6']  - mr6          # relative strength (mid)
        x[f'{X_PREFIX}relret24'] = d['r24'] - mr24         # relative strength (long)
        x[f'{X_PREFIX}relvol']   = d['atr'] / (matr + 1e-9)# relative volatility vs rest
        x[f'{X_PREFIX}relrsi']   = d['rsi'] - mrsi         # relative momentum
        x[f'{X_PREFIX}beta']     = d['r1'].rolling(60).cov(mr1) / (mr1.rolling(60).var() + 1e-9)
        x[f'{X_PREFIX}corr']     = d['r1'].rolling(60).corr(mr1)
        x[f'{X_PREFIX}rank6']    = rank6[a]                # cross-sectional rank
        bx = bases[a].set_index('timestamp')
        for col in x.columns: bx[col] = x[col]
        bases[a] = bx.reset_index()
    return bases


def add_absorption(bases):
    """Market-wide PCA absorption ratio (systemic fragility) + its momentum.
    A single value per timestamp from the BTC/ETH/SOL covariance — broadcast to all
    assets. Needs the universe even for a BTC-only model. Causal."""
    keys = list(bases.keys())
    comp = {a: bases[a].set_index('timestamp')['Close'].pct_change(1) for a in keys}
    idx  = sorted(set().union(*[comp[a].index for a in keys]))
    R    = pd.concat([comp[a].reindex(idx) for a in keys], axis=1, keys=keys)
    ab   = pd.Series(rolling_absorption(R.values, ABS_WINDOW), index=R.index)
    ab_chg = ab.diff(6)
    for a in keys:
        bx = bases[a].set_index('timestamp')
        bx[f'{X_PREFIX}absorb']     = ab.reindex(bx.index)
        bx[f'{X_PREFIX}absorb_chg'] = ab_chg.reindex(bx.index)
        bases[a] = bx.reset_index()
    return bases


print(f"BTC_ONLY:{BTC_ONLY}  AssetEmb:{'ON' if USE_ASSET_EMB else 'OFF'}  "
      f"Cross:{'ON' if USE_CROSS else 'OFF'}  NewFeat:{'ON' if USE_NEWFEAT else 'OFF'}  tag={TAG}")
print("Building BTC + ETH + SOL feature bases …")
bases = {a: build_base(f) for a, f in ASSETS.items()}
if USE_CROSS:
    bases = add_cross_asset(bases)
if USE_NEWFEAT:
    bases = add_absorption(bases)        # independent of CROSS (needs universe)
print(f"  cross-asset/absorption cols: {sum(c.startswith(X_PREFIX) for c in bases['btc'].columns)}")
RAW_COLS = {'timestamp','Open','High','Low','Close','Volume','label','taker_buy_vol','n_trades','quote_vol'}
feat_cols = [c for c in bases['btc'].columns if c not in RAW_COLS and bases['btc'][c].nunique()>1]
for a in bases: bases[a] = bases[a].dropna(subset=feat_cols).reset_index(drop=True)
print(f"  feat_cols:{len(feat_cols)}  " + "  ".join(f"{a}:{len(bases[a]):,}" for a in bases))


def labeled_positions(base):
    X = base[feat_cols].values.astype(np.float32); lab = base['label'].values; ts = base['timestamp'].values
    valid = ~np.isnan(lab)
    pos = [(i, int(lab[i-1])) for i in range(SEQ_LEN, len(X)+1) if valid[i-1]]
    return X, ts, pos

data = {a: labeled_positions(bases[a]) for a in bases}

# BTC split boundaries
Xb, tsb, posb = data['btc']
btc_end_ts = np.array([tsb[i-1] for i,_ in posb])
T_train = btc_end_ts[int(len(posb)*TRAIN_FRAC)]; T_val = btc_end_ts[int(len(posb)*VAL_FRAC)]
print(f"  T_train={pd.Timestamp(T_train):%Y-%m-%d}  T_val={pd.Timestamp(T_val):%Y-%m-%d}")

# Shared scaler — fit on pooled TRAIN-era rows of the POOL assets (BTC only if BTC_ONLY)
pool_assets = ['btc'] if BTC_ONLY else list(data.keys())
train_rows = []
for a in pool_assets:
    X,ts,pos = data[a]; b = int(np.searchsorted(ts, T_train)); train_rows.append(X[:b])
scaler = RobustScaler(quantile_range=(5,95)); scaler.fit(np.vstack(train_rows))
Xs = {a: np.clip(scaler.transform(data[a][0]), -6, 6).astype(np.float32) for a in data}

def seqs_for(asset, lo=None, hi=None):
    X, ts, pos = data[asset]; Xsc = Xs[asset]; aid = ASSET_ID[asset]
    Xo, yo, to, ao = [], [], [], []
    for i, lbl in pos:
        e = ts[i-1]
        if (lo is None or e>=lo) and (hi is None or e<hi):
            Xo.append(Xsc[i-SEQ_LEN:i]); yo.append(lbl); to.append(e); ao.append(aid)
    if not Xo:
        z=np.empty((0,SEQ_LEN,len(feat_cols)),np.float32)
        return z, np.empty((0,),np.int64), np.empty((0,),'datetime64[ns]'), np.empty((0,),np.int64)
    return np.stack(Xo).astype(np.float32), np.array(yo,np.int64), np.array(to), np.array(ao,np.int64)

# Train = pooled assets before T_train (BTC only if BTC_ONLY); Val/Test = BTC only
trX, trY, trA = [], [], []
for a in pool_assets:
    X,y,_,A = seqs_for(a, hi=T_train); trX.append(X); trY.append(y); trA.append(A)
Xtr=np.concatenate(trX); ytr=np.concatenate(trY); atr_id=np.concatenate(trA)
Xva, yva, _, _      = seqs_for('btc', lo=T_train, hi=T_val)
Xte, yte, ts_te, _  = seqs_for('btc', lo=T_val)
vol_idx = feat_cols.index(VOL_FEATURE)
atr_tr_btc = Xtr[atr_id==ASSET_ID['btc']][:, -1, vol_idx]
counts = Counter(ytr.tolist())
per_asset = {a: int((atr_id==ASSET_ID[a]).sum()) for a in ASSET_ID}
print(f"  train pooled:{len(ytr):,}  ({per_asset})   Short:{counts[0]:,} Long:{counts[1]:,}")
print(f"  val(btc):{len(yva):,}  test(btc):{len(yte):,}")


class SeqDS(Dataset):
    def __init__(self, X, y, A, augment=False):
        self.X=torch.from_numpy(X.astype(np.float32)); self.y=torch.from_numpy(y.astype(np.int64))
        self.A=torch.from_numpy(A.astype(np.int64)); self.augment=augment
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        x=self.X[i].clone()
        if self.augment:
            x=x+AUG_NOISE*torch.randn_like(x); m=torch.bernoulli(torch.full((x.shape[-1],),AUG_MASK_P)).bool(); x[:,m]=0.0
        return x, self.y[i], self.A[i]

train_dl = DataLoader(SeqDS(Xtr,ytr,atr_id,augment=True), batch_size=BATCH, shuffle=True, drop_last=True)
val_dl   = DataLoader(SeqDS(Xva,yva,np.zeros(len(yva),np.int64)), batch_size=BATCH, shuffle=False)
test_dl  = DataLoader(SeqDS(Xte,yte,np.zeros(len(yte),np.int64)), batch_size=BATCH, shuffle=False)


# ── MODEL (transformer + optional asset embedding) ─────────────────
class SqueezeExcite(nn.Module):
    def __init__(self,n_feat,r=4):
        super().__init__(); self.fc=nn.Sequential(nn.Linear(n_feat,max(n_feat//r,16)),nn.ReLU(),nn.Linear(max(n_feat//r,16),n_feat),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x.mean(1)).unsqueeze(1)
class PatchEmbed(nn.Module):
    def __init__(self,n_feat,patch=PATCH_SIZE,d=D_MODEL):
        super().__init__(); self.p=patch; self.proj=nn.Linear(n_feat*patch,d); self.norm=nn.LayerNorm(d)
    def forward(self,x):
        B,T,Fd=x.shape; pad=(self.p-T%self.p)%self.p
        if pad: x=F.pad(x,(0,0,0,pad))
        return self.norm(self.proj(x.reshape(B,-1,self.p*Fd)))
class RoPEEmbedding(nn.Module):
    def __init__(self,dim,base=10000):
        super().__init__(); inv=1.0/(base**(torch.arange(0,dim,2).float()/dim)); self.register_buffer('inv_freq',inv)
    def forward(self,L,device):
        t=torch.arange(L,device=device).float(); fr=torch.outer(t,self.inv_freq); emb=torch.cat([fr,fr],-1); return emb.cos(),emb.sin()
def _rotate_half(x): h=x.shape[-1]//2; return torch.cat([-x[...,h:],x[...,:h]],-1)
def _apply_rope(q,k,cos,sin): cos=cos[None,None];sin=sin[None,None]; return q*cos+_rotate_half(q)*sin,k*cos+_rotate_half(k)*sin
def _alibi_slopes(nh):
    def sl(n): return [2**(-8*i/n) for i in range(1,n+1)]
    if (nh&(nh-1))==0: return torch.tensor(sl(nh),dtype=torch.float32)
    p=2**int(np.floor(np.log2(nh))); return torch.tensor(sl(p)+sl(2*p)[0::2][:nh-p],dtype=torch.float32)
def _alibi_bias(nh,L,device):
    s=_alibi_slopes(nh).to(device); dist=(torch.arange(L,device=device).unsqueeze(0)-torch.arange(L,device=device).unsqueeze(1)).abs().float()
    return -s.view(-1,1,1)*dist.unsqueeze(0)
class RelativeAttention(nn.Module):
    def __init__(self,d_model=D_MODEL,n_heads=N_HEADS,dropout=DROPOUT):
        super().__init__(); self.H=n_heads; self.dh=d_model//n_heads; self.scl=self.dh**-0.5
        self.qkv=nn.Linear(d_model,3*d_model,bias=False); self.out=nn.Linear(d_model,d_model,bias=False)
        self.drop=nn.Dropout(dropout); self.rope=RoPEEmbedding(self.dh)
    def forward(self,x,return_attn=False):
        B,L,D=x.shape; H,dh=self.H,self.dh
        Q,K,V=self.qkv(x).chunk(3,-1)
        Q=Q.view(B,L,H,dh).transpose(1,2); K=K.view(B,L,H,dh).transpose(1,2); V=V.view(B,L,H,dh).transpose(1,2)
        cos,sin=self.rope(L,x.device); Q,K=_apply_rope(Q,K,cos,sin)
        logits=(Q@K.transpose(-2,-1))*self.scl+_alibi_bias(H,L,x.device)
        attn_w=logits.softmax(-1)
        out=self.out((self.drop(attn_w)@V).transpose(1,2).reshape(B,L,D))
        return (out, attn_w.detach()) if return_attn else out
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.n1=nn.LayerNorm(D_MODEL); self.attn=RelativeAttention(); self.n2=nn.LayerNorm(D_MODEL)
        self.ff=nn.Sequential(nn.Linear(D_MODEL,D_FF),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(D_FF,D_MODEL),nn.Dropout(DROPOUT))
    def forward(self,x,return_attn=False):
        if return_attn:
            ao,aw=self.attn(self.n1(x),return_attn=True); x=x+ao; x=x+self.ff(self.n2(x)); return x,aw
        x=x+self.attn(self.n1(x)); x=x+self.ff(self.n2(x)); return x
class TemporalTransformer(nn.Module):
    def __init__(self,n_feat,n_assets=N_ASSETS,use_emb=USE_ASSET_EMB):
        super().__init__(); self.se=SqueezeExcite(n_feat); self.embed=PatchEmbed(n_feat)
        self.blocks=nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.use_emb=use_emb
        if use_emb: self.asset_emb=nn.Embedding(n_assets, D_MODEL)
        self.head=nn.Sequential(nn.LayerNorm(D_MODEL),nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(D_MODEL//2,2))
    def forward(self,x,aid=None,return_attn=False):
        x=self.se(x); tok=self.embed(x)
        if return_attn:
            attns=[]
            for blk in self.blocks: tok,aw=blk(tok,return_attn=True); attns.append(aw)
            pooled=tok.mean(1)
            if self.use_emb and aid is not None: pooled=pooled+self.asset_emb(aid)
            return self.head(pooled), attns
        for blk in self.blocks: tok=blk(tok)
        pooled=tok.mean(1)
        if self.use_emb and aid is not None: pooled=pooled+self.asset_emb(aid)
        return self.head(pooled)


class FocalLS(nn.Module):
    def __init__(self, gamma=2.0, smooth=LABEL_SMOOTH, weight=None):
        super().__init__(); self.gamma=gamma; self.smooth=smooth; self.weight=weight
    def forward(self, logits, targets):
        n=logits.size(1)
        with torch.no_grad():
            y_sm=torch.full_like(logits,self.smooth/(n-1)); y_sm.scatter_(1,targets.unsqueeze(1),1.0-self.smooth)
        log_p=F.log_softmax(logits,1); ce=-(y_sm*log_p).sum(1)
        pt=torch.exp(-F.cross_entropy(logits,targets,weight=self.weight,reduction='none'))
        return (((1-pt)**self.gamma)*ce).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
n_feat = Xtr.shape[2]
model = TemporalTransformer(n_feat).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}  (asset_emb={'on' if USE_ASSET_EMB else 'off'})")
total=sum(counts.values())
w=torch.tensor([total/(2*counts[0])*SHORT_WEIGHT_FACTOR, total/(2*counts[1])],dtype=torch.float32).to(device)
criterion=FocalLS(gamma=2.0,smooth=LABEL_SMOOTH,weight=w)
optimizer=torch.optim.AdamW(model.parameters(),lr=LR_MAX,weight_decay=1e-3)
def lr_lambda(ep):
    if ep<WARMUP_EP: return (ep+1)/WARMUP_EP
    p=(ep-WARMUP_EP)/max(1,EPOCHS-WARMUP_EP); return 0.5*(1+np.cos(np.pi*p))
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

best_val,best_state,no_imp=float('inf'),None,0
hist={'tl':[],'vl':[],'va':[]}
print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}")
print("-"*48)
for ep in range(1,EPOCHS+1):
    model.train(); tl=0.0
    for xb,yb,ab in train_dl:
        xb,yb,ab=xb.to(device),yb.to(device),ab.to(device)
        optimizer.zero_grad(); loss=criterion(model(xb,ab),yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); tl+=loss.item()
    tl/=len(train_dl); scheduler.step()
    model.eval(); vl,vp,vt=0.0,[],[]
    with torch.no_grad():
        for xb,yb,ab in val_dl:
            xb,yb,ab=xb.to(device),yb.to(device),ab.to(device); out=model(xb,ab)
            vl+=criterion(out,yb).item(); vp.extend(out.argmax(1).cpu().tolist()); vt.extend(yb.cpu().tolist())
    vl/=len(val_dl); va=accuracy_score(vt,vp)
    hist['tl'].append(tl); hist['vl'].append(vl); hist['va'].append(va)
    if vl<best_val:
        best_val=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0; mk=' *'
    else: no_imp+=1; mk=f'  (no improve {no_imp}/{PATIENCE})'
    print(f"{ep:4d}  {tl:8.4f}  {vl:8.4f}  {va:7.4f}  {optimizer.param_groups[0]['lr']:9.2e}{mk}")
    if no_imp>=PATIENCE: print(f"  → Early stop at epoch {ep}"); break
print(f"\nBest val loss (BTC): {best_val:.4f}")

model.load_state_dict(best_state); model.eval()
tpreds,tprobs,ttrue=[],[],[]
with torch.no_grad():
    for xb,yb,ab in test_dl:
        out=model(xb.to(device),ab.to(device)); prbs=torch.softmax(out,1).cpu().numpy()
        tpreds.extend(out.argmax(1).cpu().tolist()); tprobs.extend(prbs.tolist()); ttrue.extend(yb.tolist())
tpreds=np.array(tpreds); tprobs=np.array(tprobs); ttrue=np.array(ttrue); p_long=tprobs[:,1]
p33,p67=np.percentile(atr_tr_btc,[33,67])
vol_te=Xte[:len(tpreds),-1,vol_idx]
long_thr=np.where(vol_te>p67,THRESH_HIGH_VOL[0],np.where(vol_te<p33,THRESH_LOW_VOL[0],LONG_THRESH))
short_thr=np.where(vol_te>p67,THRESH_HIGH_VOL[1],np.where(vol_te<p33,THRESH_LOW_VOL[1],SHORT_THRESH))
regime=np.where(vol_te>p67,'high',np.where(vol_te<p33,'low','mid'))
sig=np.where(p_long>=long_thr,1,np.where(p_long<=short_thr,0,-1)); mask=sig!=-1
print("\n"+"="*55); print(f"BTC+ETH+SOL POOLED ({TAG}) — BTC TEST (argmax)"); print("="*55)
print(f"Accuracy: {accuracy_score(ttrue,tpreds):.4f}")
print(classification_report(ttrue,tpreds,target_names=['Short','Long']))
print("="*55); print("ADAPTIVE CONFIDENCE-FILTERED SIGNALS (BTC test)"); print("="*55)
print(f"  LONG:{(sig==1).sum():,}  SHORT:{(sig==0).sum():,}  NEUTRAL:{(sig==-1).sum():,} ({100*(sig==-1).sum()/len(sig):.1f}% filtered)")
if mask.sum()>0:
    print(f"\nAccuracy on active signals: {accuracy_score(ttrue[mask],sig[mask]):.4f}")
    print(classification_report(ttrue[mask],sig[mask],target_names=['Short','Long']))
    print("Confusion matrix:"); print(confusion_matrix(ttrue[mask],sig[mask]))
print(f"\nProb(Long) — mean:{p_long.mean():.3f}  std:{p_long.std():.3f}  min:{p_long.min():.3f}  max:{p_long.max():.3f}")

ckpt=os.path.join(SAVE_DIR,f'btc_eth_sol_pooled_model_{TAG}.pth')
torch.save({'model_state':best_state,
            'model_cfg':dict(n_feat=n_feat,d_model=D_MODEL,n_heads=N_HEADS,n_layers=N_LAYERS,d_ff=D_FF,drop=DROPOUT,
                             patch_size=PATCH_SIZE,n_assets=N_ASSETS,use_emb=USE_ASSET_EMB),
            'scaler':scaler,'feat_cols':feat_cols,'seq_len':SEQ_LEN,'history':hist,
            'vol_p33':float(p33),'vol_p67':float(p67)}, ckpt)
print(f"\nCheckpoint → {ckpt}")
fig,ax=plt.subplots(1,2,figsize=(13,4))
ax[0].plot(hist['tl'],label='Train(pooled)'); ax[0].plot(hist['vl'],label='Val(BTC)')
ax[0].set_title(f'BTC+ETH+SOL ({TAG}) Loss'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(hist['va'],color='green'); ax[1].axhline(0.5,color='gray',ls='--'); ax[1].set_title('Val Acc (BTC)'); ax[1].grid(True)
plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR,f'sol_pooled_curves_{TAG}.png'),dpi=150); plt.close()
sig_df=pd.DataFrame({'timestamp':ts_te[:len(tpreds)],'true_label':ttrue,'prob_short':tprobs[:,0],'prob_long':p_long,
                     'vol_regime':regime,'signal':np.where(sig==1,'LONG',np.where(sig==0,'SHORT','NEUTRAL'))})
sig_df.to_csv(os.path.join(SAVE_DIR,f'btc_eth_sol_pooled_signals_{TAG}.csv'),index=False)
print(f"Signals → btc_eth_sol_pooled_signals_{TAG}.csv  ({(sig_df.signal!='NEUTRAL').sum():,} active)")


# ═══════════════════════════════════════════════════════════════════
# ATTENTION MAP ANALYSIS  (same as baseline; BTC test set)
# ═══════════════════════════════════════════════════════════════════
# attn[n,l,h,i,j] = query-patch i attends to key-patch j (softmax over j → rows sum to 1)
# "received attention" for patch j = column-sum over queries. Uniform baseline = 1.0.
print("\n" + "="*55); print("ATTENTION MAP ANALYSIS"); print("="*55)
model.eval()
N_PATCHES = SEQ_LEN // PATCH_SIZE
attn_store = []
with torch.no_grad():
    for xb, _, ab in test_dl:
        _, layer_attns = model(xb.to(device), ab.to(device), return_attn=True)
        attn_store.append(torch.stack(layer_attns, dim=1).cpu().numpy())   # (B, Nl, H, Np, Np)
attn_all = np.concatenate(attn_store, axis=0)
recv = attn_all.sum(axis=-2).mean(axis=(1, 2))                              # (N_test, Np)
patch_pos = np.arange(N_PATCHES)

# 1. Recency profile
print(f"\nReceived attention per patch  (uniform baseline = 1.000):")
print(f"  {'p':>2}  [steps]  recv    bar")
for p in range(N_PATCHES):
    val = recv[:, p].mean(); bar = '█' * int(val * 16)
    tag = '  ← most recent' if p == N_PATCHES - 1 else ''
    print(f"  {p:2d}  [{p*PATCH_SIZE:2d}–{p*PATCH_SIZE+PATCH_SIZE-1:2d}]  {val:.3f}  {bar}{tag}")
early = recv[:, :4].mean(); late = recv[:, -4:].mean()
print(f"\nEarly patches [0–3]   mean received: {early:.4f}")
print(f"Late  patches [12–15] mean received: {late:.4f}")
print(f"Recency ratio (late / early)        : {late/early:.2f}×")

# 2. Long vs Short attention profile
long_m = sig == 1; short_m = sig == 0
if long_m.sum() and short_m.sum():
    print(f"\nRecent-patch attention (last 4 patches):")
    print(f"  Long  signals (n={long_m.sum():,}): {recv[long_m, -4:].mean():.4f}")
    print(f"  Short signals (n={short_m.sum():,}): {recv[short_m, -4:].mean():.4f}")
    print(f"  Early-patch attention (first 4 patches):")
    print(f"  Long  signals: {recv[long_m, :4].mean():.4f}")
    print(f"  Short signals: {recv[short_m, :4].mean():.4f}")
    if recv[short_m, :4].mean() > recv[long_m, :4].mean():
        print("  → Short signals draw relatively more attention from older patches")

# 3. Per-layer entropy
print(f"\nPer-layer attention entropy  (lower = more focused):")
for l in range(N_LAYERS):
    a = attn_all[:, l].mean(axis=(0, 1))
    ent = -(a * np.log(a + 1e-9)).sum(axis=-1).mean()
    print(f"  Layer {l+1}: {ent:.3f}  {'░' * int(ent * 8)}")

# 4. OFI-divergence vs recent attention
ofi_idx = feat_cols.index('4h_ofi_div') if '4h_ofi_div' in feat_cols else None
if ofi_idx is not None:
    ofi_vals = Xte[:len(sig), -1, ofi_idx]; recent_attn = recv[:, -4:].mean(axis=1)
    print(f"\nOFI-divergence vs recent-patch attention:")
    for lbl, m in [('Long', long_m), ('Short', short_m)]:
        if m.sum() > 5:
            r = np.corrcoef(np.abs(ofi_vals[m]), recent_attn[m])[0, 1]
            print(f"  {lbl:5s} (n={m.sum():,}): r = {r:+.3f}")

# 5. Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
for lbl, dat, col, ls in [('All test', recv, 'gray', '-'),
                          ('Long (active)', recv[long_m], 'steelblue', '-'),
                          ('Short (active)', recv[short_m], 'tomato', '-'),
                          ('Correct', recv[(sig != -1) & (sig == ttrue)], 'green', '--'),
                          ('Wrong', recv[(sig != -1) & (sig != ttrue)], 'orange', '--')]:
    if len(dat): ax.plot(patch_pos, dat.mean(0), marker='o', markersize=4, label=f'{lbl} (n={len(dat)})', color=col, linestyle=ls)
ax.axvspan(11.5, 15.5, alpha=0.08, color='green'); ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, label='uniform')
ax.set_xlabel('Patch index (0=oldest · 15=most recent)'); ax.set_ylabel('Mean received attention')
ax.set_title('Received attention per patch — by signal type'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax = axes[0, 1]
heat = attn_all[:, -1].mean(axis=(0, 1))
im = ax.imshow(heat, cmap='plasma', aspect='auto', origin='upper')
ax.set_xlabel('Key patch'); ax.set_ylabel('Query patch'); ax.set_title(f'Layer {N_LAYERS} mean attention map'); plt.colorbar(im, ax=ax)
ax = axes[1, 0]
for l in range(N_LAYERS):
    ax.plot(patch_pos, attn_all[:, l].sum(axis=-2).mean(axis=(0, 1)), marker='o', markersize=3, label=f'Layer {l+1}')
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Patch index'); ax.set_ylabel('Mean received attention'); ax.set_title('Received attention by layer'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax = axes[1, 1]
for lbl, m, col, mk in [('Long correct', (sig==1)&(sig==ttrue), 'steelblue', 'o'),
                        ('Long wrong', (sig==1)&(sig!=ttrue), 'lightblue', 's'),
                        ('Short correct', (sig==0)&(sig==ttrue), 'tomato', 'o'),
                        ('Short wrong', (sig==0)&(sig!=ttrue), 'salmon', 's')]:
    if m.sum() > 0: ax.plot(patch_pos, recv[m].mean(0), marker=mk, markersize=4, label=f'{lbl} n={m.sum()}', color=col)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Patch index'); ax.set_ylabel('Mean received attention'); ax.set_title('Attention — correct vs wrong'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle(f'BTC+ETH+SOL ({TAG}) — Attention Map Analysis', fontsize=12, fontweight='bold')
plt.tight_layout()
attn_path = os.path.join(SAVE_DIR, f'attention_analysis_{TAG}.png')
plt.savefig(attn_path, dpi=150); plt.close()
print(f"\nAttention maps → {attn_path}")
