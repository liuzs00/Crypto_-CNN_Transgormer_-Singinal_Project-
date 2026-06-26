"""
crypto_features.py — Shared feature pipeline + config for the production line.

ONE source of truth for feature construction so the trainer, backtest and predict scripts
build byte-identical inputs (no "must match trainer" copies to drift). All features are
scale-invariant + strictly causal; see FEATURES.md for the dictionary.

Config is env-driven (so the experiment A/B protocol keeps working), but the DEFAULTS are the
production config — importing this module with no env vars gives the 282-feature production
set (NEWFEAT/CROSS/CONVSTEM on), so the backtest/predict reconstruct production features
without setting anything. Ablations flip the flags (e.g. NEWFEAT=0, CONVSTEM=0).
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = r"D:\Document\LLLLLLLLLLLLL\DATA"
SEQ_LEN  = 64

# triple-barrier (per-direction): each side resolved against its own TP/SL
LONG_TP, LONG_SL, SHORT_TP, SHORT_SL, TB_TIMEOUT = 0.03, 0.015, 0.03, 0.015, 4

# ── experiment / feature flags (defaults = production) ──
BTC_ONLY      = os.environ.get('BTC_ONLY', '0') == '1'        # train on BTC rows only (ablation)
USE_ASSET_EMB = (os.environ.get('ASSET_EMB', '1') == '1') and not BTC_ONLY
USE_CROSS     = os.environ.get('CROSS', '1') == '1'          # cross-asset (BTC vs ETH/SOL) features
USE_NEWFEAT   = os.environ.get('NEWFEAT', '1') == '1'        # Anchored-VWAP + Absorption + path-entropy (PRODUCTION default ON)
USE_HAWKES    = os.environ.get('HAWKES', '0') == '1'         # rejected experiment (kept default-off)
HAWKES_W      = 20
USE_DWT       = os.environ.get('DWT', '0') == '1'            # rejected experiment (kept default-off)
USE_ENTROPY   = os.environ.get('ENTROPY', '0') == '1'       # path entropy (also folded into NEWFEAT)
ENT_W         = 50
USE_CONVSTEM  = os.environ.get('CONVSTEM', '1') == '1'      # temporal conv-stem tokenizer (production default)
ABS_WINDOW    = 60
X_PREFIX      = 'x_'
RAW_COLS      = {'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'label',
                 'taker_buy_vol', 'n_trades', 'quote_vol'}

ASSETS = {
    'btc': {'15m': 'btc_15m_data_2018_to_2025.csv', '1h': 'btc_1h_data_2018_to_2025.csv',
            '4h': 'btc_4h_data_2018_to_2025.csv', '1d': 'btc_1d_data_2018_to_2025.csv'},
    'eth': {'15m': 'eth_15m_data_2018_to_2025.csv', '1h': 'eth_1h_data_2018_to_2025.csv',
            '4h': 'eth_4h_data_2018_to_2025.csv', '1d': 'eth_1d_data_2018_to_2025.csv'},
    'sol': {'15m': 'sol_15m_data_2018_to_2025.csv', '1h': 'sol_1h_data_2018_to_2025.csv',
            '4h': 'sol_4h_data_2018_to_2025.csv', '1d': 'sol_1d_data_2018_to_2025.csv'},
}
ASSET_ID = {'btc': 0, 'eth': 1, 'sol': 2}
N_ASSETS = len(ASSET_ID)

TAG = ('btconly' if BTC_ONLY else ('emb' if USE_ASSET_EMB else 'noemb')) + \
      ('_cross' if USE_CROSS else '') + ('_nf' if USE_NEWFEAT else '') + \
      ('_hwk' if USE_HAWKES else '') + ('_dwt' if USE_DWT else '') + ('_ent' if USE_ENTROPY else '') + \
      ('' if USE_CONVSTEM else '_linpatch')


def load_csv(path):
    wanted = {'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Number of trades', 'Quote asset volume'}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    df = df.rename(columns={'Open time': 'timestamp', 'Taker buy base asset volume': 'taker_buy_vol',
                            'Number of trades': 'n_trades', 'Quote asset volume': 'quote_vol'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col != 'timestamp': df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)


def atr_ema(h, l, c, period=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
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
        if USE_HAWKES:
            ntm=nt.rolling(HAWKES_W).mean(); ntv=nt.rolling(HAWKES_W).var()
            eta=(1-np.sqrt((ntm/(ntv+1e-9)).clip(lower=0))).clip(lower=0,upper=0.99)
            d[f'{px}hawkes_eta']=eta
            d[f'{px}hawkes_eta_mom']=eta-eta.shift(6)
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
    # ── Discrete-wavelet detail coefficients (rejected experiment; default off) ──
    if USE_DWT:
        lc=np.log(c.clip(lower=1e-9))
        s1=lc.rolling(2).mean(); s2=lc.rolling(4).mean(); s3=lc.rolling(8).mean()
        d2=s1-s2; d3=s2-s3
        d[f'{px}dwt_d2']=d2; d[f'{px}dwt_d3']=d3
        d[f'{px}dwt_d2_e']=np.sqrt((d2**2).rolling(6).mean())
        d[f'{px}dwt_d3_e']=np.sqrt((d3**2).rolling(12).mean())
    # ── Path entropy: permutation entropy (Bandt-Pompe, order d=3) on LOG-PRICE ──
    if USE_NEWFEAT or USE_ENTROPY:   # adopted into the production feature set (282 feat)
        _x=np.log(c.clip(lower=1e-9)).values; _a=np.roll(_x,2); _b=np.roll(_x,1)
        code=((_a<_b).astype(np.int64)*4+(_a<_x).astype(np.int64)*2+(_b<_x).astype(np.int64))
        code[:2]=-1
        oh=np.zeros((len(_x),8)); ok=code>=0
        oh[np.arange(len(_x))[ok], code[ok]]=1.0
        cnt=pd.DataFrame(oh).rolling(ENT_W).sum().values
        pp=cnt/(cnt.sum(1,keepdims=True)+1e-9)
        with np.errstate(divide='ignore',invalid='ignore'):
            ent=-(np.where(pp>0, pp*np.log(pp), 0.0)).sum(1)/np.log(6)
        d[f'{px}pent']=ent
        d[f'{px}pent_chg']=pd.Series(ent,index=d.index)-pd.Series(ent,index=d.index).shift(6)
    # ── Anchored-VWAP distance (z-score vs institutional cost basis) ──
    if USE_NEWFEAT and 'timestamp' in d.columns:
        ts = pd.to_datetime(d['timestamp']); iso = ts.dt.isocalendar()
        wk = (iso.year*100 + iso.week).values; mo = (ts.dt.year*100 + ts.dt.month).values
        pv = ((h+l+c)/3*v)
        gw = pd.DataFrame({'pv':pv.values,'v':v.values,'k':wk})
        avwap_w = (gw.groupby('k')['pv'].cumsum()/(gw.groupby('k')['v'].cumsum()+1e-9)).values
        gm = pd.DataFrame({'pv':pv.values,'v':v.values,'k':mo})
        avwap_m = (gm.groupby('k')['pv'].cumsum()/(gm.groupby('k')['v'].cumsum()+1e-9)).values
        d[f'{px}vwap_dist_w'] = (c.values - avwap_w)/(atr14.values+1e-9)
        d[f'{px}vwap_dist_m'] = (c.values - avwap_m)/(atr14.values+1e-9)
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


# ── cross-asset (each asset vs the REST) + systemic absorption ──
def rolling_absorption(R, window):
    """Absorption ratio = variance share of the 1st PC over a rolling window. Causal."""
    T = R.shape[0]; out = np.full(T, np.nan)
    for t in range(window, T):
        win = R[t-window:t]; m = ~np.isnan(win).any(axis=0); w = win[:, m]
        if w.shape[1] < 2: continue
        ev = np.linalg.eigvalsh(np.cov(w, rowvar=False)); ev = ev[ev > 0]
        if ev.size: out[t] = ev.max() / ev.sum()
    return out


def add_cross_asset(bases):
    keys = list(bases.keys())
    comp = {}
    for a in keys:
        b = bases[a].set_index('timestamp'); c = b['Close']
        comp[a] = pd.DataFrame({
            'r1': c.pct_change(1), 'r6': c.pct_change(6), 'r24': c.pct_change(24),
            'atr': b['4h_atr14'], 'rsi': b['4h_rsi14'],
        })
    idx = sorted(set().union(*[comp[a].index for a in keys]))
    for a in keys: comp[a] = comp[a].reindex(idx)
    rank6 = pd.concat([comp[a]['r6'] for a in keys], axis=1, keys=keys).rank(axis=1, pct=True)
    for a in keys:
        others = [k for k in keys if k != a]
        mr1  = pd.concat([comp[k]['r1']  for k in others], axis=1).mean(axis=1)
        mr6  = pd.concat([comp[k]['r6']  for k in others], axis=1).mean(axis=1)
        mr24 = pd.concat([comp[k]['r24'] for k in others], axis=1).mean(axis=1)
        matr = pd.concat([comp[k]['atr'] for k in others], axis=1).mean(axis=1)
        mrsi = pd.concat([comp[k]['rsi'] for k in others], axis=1).mean(axis=1)
        d = comp[a]; x = pd.DataFrame(index=idx)
        x[f'{X_PREFIX}relret1']  = d['r1']  - mr1
        x[f'{X_PREFIX}relret6']  = d['r6']  - mr6
        x[f'{X_PREFIX}relret24'] = d['r24'] - mr24
        x[f'{X_PREFIX}relvol']   = d['atr'] / (matr + 1e-9)
        x[f'{X_PREFIX}relrsi']   = d['rsi'] - mrsi
        x[f'{X_PREFIX}beta']     = d['r1'].rolling(60).cov(mr1) / (mr1.rolling(60).var() + 1e-9)
        x[f'{X_PREFIX}corr']     = d['r1'].rolling(60).corr(mr1)
        x[f'{X_PREFIX}rank6']    = rank6[a]
        bx = bases[a].set_index('timestamp')
        for col in x.columns: bx[col] = x[col]
        bases[a] = bx.reset_index()
    return bases


def add_absorption(bases):
    """Market-wide PCA absorption ratio (systemic fragility) + its momentum. Causal."""
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


def build_bases(apply_cross=None, apply_absorb=None):
    """Build the full per-asset feature bases for the production model. Returns {asset: df}.
    apply_cross/apply_absorb default to the module flags (USE_CROSS / USE_NEWFEAT)."""
    ac = USE_CROSS   if apply_cross  is None else apply_cross
    aa = USE_NEWFEAT if apply_absorb is None else apply_absorb
    bases = {a: build_base(f) for a, f in ASSETS.items()}
    if ac: bases = add_cross_asset(bases)
    if aa: bases = add_absorption(bases)       # absorption needs the whole universe
    return bases


def select_feat_cols(bases):
    """Frozen, ordered feature columns: non-raw, non-constant columns of the BTC base."""
    return [c for c in bases['btc'].columns if c not in RAW_COLS and bases['btc'][c].nunique() > 1]
