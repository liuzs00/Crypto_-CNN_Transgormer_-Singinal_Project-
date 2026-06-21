"""
btc_backtest_cross_nf.py — Backtest the newest model
btc_eth_sol_pooled_model_emb_cross_nf.pth (BTC+ETH+SOL pooled transformer +
asset embedding + crypto cross-asset features + Anchored-VWAP + Absorption-Ratio).

Rebuilds the FULL 274-feature pipeline for BTC inference (needs ETH/SOL for the
cross-asset + absorption features), runs the asset-embedding model with asset_id=0,
then reuses btc_backtest.py's triple-barrier trade simulation + metrics.

Run:  py -3.10 btc_backtest_cross_nf.py            # BTC test split (default)
      py -3.10 btc_backtest_cross_nf.py --split full
"""
import os, sys, argparse, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings('ignore')
import btc_backtest as B          # simulate_trade, run_trades, equity_curve, stats, max_drawdown, costs
import btc_predict as P           # THRESH_*, VOL_FEATURE

DATA_DIR  = r"D:\Document\LLLLLLLLLLLLL\DATA"
SAVE_DIR  = r"D:\Document\LLLLLLLLLLLLL"
CKPT_PATH = os.environ.get('CKPT', os.path.join(SAVE_DIR, 'btc_eth_sol_pooled_model_emb_cross_nf.pth'))
SEQ_LEN   = 64; PATCH_SIZE = 4; ABS_WINDOW = 60
LONG_TP, LONG_SL, SHORT_TP, SHORT_SL, TB_TIMEOUT = 0.03, 0.015, 0.03, 0.015, 4
TRAIN_FRAC, VAL_FRAC = 0.70, 0.85
X_PREFIX = 'x_'
D_MODEL=256; N_HEADS=8; N_LAYERS=3; D_FF=512; DROPOUT=0.25
ASSETS = {
    'btc': {'15m':'btc_15m_data_2018_to_2025.csv','1h':'btc_1h_data_2018_to_2025.csv','4h':'btc_4h_data_2018_to_2025.csv','1d':'btc_1d_data_2018_to_2025.csv'},
    'eth': {'15m':'eth_15m_data_2018_to_2025.csv','1h':'eth_1h_data_2018_to_2025.csv','4h':'eth_4h_data_2018_to_2025.csv','1d':'eth_1d_data_2018_to_2025.csv'},
    'sol': {'15m':'sol_15m_data_2018_to_2025.csv','1h':'sol_1h_data_2018_to_2025.csv','4h':'sol_4h_data_2018_to_2025.csv','1d':'sol_1d_data_2018_to_2025.csv'},
}


# ── feature pipeline (verbatim from btc_eth_sol_cross_train.py, NEWFEAT on) ──
def load_csv(path):
    wanted={'Open time','Open','High','Low','Close','Volume','Taker buy base asset volume','Number of trades','Quote asset volume'}
    df=pd.read_csv(path, usecols=lambda c: c in wanted)
    df=df.rename(columns={'Open time':'timestamp','Taker buy base asset volume':'taker_buy_vol','Number of trades':'n_trades','Quote asset volume':'quote_vol'})
    df['timestamp']=pd.to_datetime(df['timestamp'],utc=True,errors='coerce'); df=df.dropna(subset=['timestamp'])
    for col in df.columns:
        if col!='timestamp': df[col]=pd.to_numeric(df[col],errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

def atr_ema(h,l,c,period=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,min_periods=period).mean()

def add_indicators(df, px=''):
    d=df.copy(); c,h,l,v,o=d['Close'],d['High'],d['Low'],d['Volume'],d['Open']
    atr14=atr_ema(h,l,c,14); atr7=atr_ema(h,l,c,7); ret1=c.pct_change()
    if 'taker_buy_vol' in d.columns:
        buy_vol=d['taker_buy_vol']; sell_vol=v-buy_vol; buy_r=buy_vol/(v+1e-9); ofi=buy_vol-sell_vol
        d[f'{px}buy_ratio']=buy_r; d[f'{px}buy_ratio_ma']=buy_r.rolling(10).mean(); d[f'{px}buy_ratio_dev']=buy_r-buy_r.rolling(20).mean()
        d[f'{px}delta_vol']=2*buy_r-1; d[f'{px}delta_vol_ma']=d[f'{px}delta_vol'].rolling(10).mean()
        for w in [5,10,20]: d[f'{px}ofi{w}']=ofi.rolling(w).sum()/(v.rolling(w).sum()+1e-9)
        d[f'{px}ofi_mom']=d[f'{px}ofi5']-d[f'{px}ofi20']
        for w in [5,20]: d[f'{px}vpin{w}']=ofi.abs().rolling(w).sum()/(v.rolling(w).sum()+1e-9)
        d[f'{px}ofi_div']=(np.sign(ret1)-np.sign(ofi)).rolling(10).mean()
        if 'quote_vol' in d.columns:
            am=ret1.abs()/(d['quote_vol']+1e-9); d[f'{px}amihud']=am/(am.rolling(20).mean()+1e-9)
    if 'n_trades' in d.columns:
        nt=d['n_trades']; d[f'{px}trade_int']=nt/(nt.rolling(20).mean()+1e-9); at=v/(nt+1e-9); d[f'{px}trade_size']=at/(at.rolling(20).mean()+1e-9)
    vs=(h-l)/(v+1e-9); d[f'{px}vol_sync']=vs/(vs.rolling(20).mean()+1e-9)
    rv=(ret1**2).rolling(20).sum(); rbv=(ret1.abs()*ret1.shift(1).abs()).rolling(20).sum()*(np.pi/2); d[f'{px}jump_ratio']=rv/(rbv+1e-9)
    vsma=v.rolling(20).mean(); vr=v/(vsma+1e-9); uw=(h-c.clip(lower=o))/(atr14+1e-9); lw=(c.clip(upper=o)-l)/(atr14+1e-9)
    d[f'{px}upper_sweep']=uw*vr; d[f'{px}lower_sweep']=lw*vr; d[f'{px}range_eff']=ret1.abs()/((h-l)/c+1e-9)
    ema9=c.ewm(span=9,min_periods=9).mean(); ema21=c.ewm(span=21,min_periods=21).mean(); ema50=c.ewm(span=50,min_periods=50).mean(); ema200=c.ewm(span=200,min_periods=200).mean()
    d[f'{px}pr9']=c/ema9-1; d[f'{px}pr21']=c/ema21-1; d[f'{px}pr50']=c/ema50-1; d[f'{px}pr200']=c/ema200-1
    d[f'{px}ema9_21']=ema9/ema21-1; d[f'{px}ema21_50']=ema21/ema50-1; d[f'{px}ema50_200']=ema50/ema200-1
    delta=c.diff()
    for per,name in [(7,'rsi7'),(14,'rsi14')]:
        g=delta.clip(lower=0).ewm(alpha=1/per,min_periods=per).mean(); ls=(-delta.clip(upper=0)).ewm(alpha=1/per,min_periods=per).mean(); d[f'{px}{name}']=(100-100/(1+g/(ls+1e-9)))/100
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean(); macd=(ema12-ema26)/(atr14+1e-9); macds=macd.ewm(span=9).mean()
    d[f'{px}macd']=macd; d[f'{px}macds']=macds; d[f'{px}macdh']=macd-macds
    d[f'{px}atr14']=atr14/(c+1e-9); d[f'{px}atr7']=atr7/(c+1e-9); d[f'{px}atr_r']=atr7/(atr14+1e-9)
    bm=c.rolling(20).mean(); bstd=c.rolling(20).std(); bup=bm+2*bstd; bdn=bm-2*bstd; d[f'{px}bbw']=(bup-bdn)/(bm+1e-9); d[f'{px}bbp']=(c-bdn)/(bup-bdn+1e-9)
    ll14=l.rolling(14).min(); hh14=h.rolling(14).max(); k=(c-ll14)/(hh14-ll14+1e-9); d[f'{px}stk']=k; d[f'{px}std']=k.rolling(3).mean(); d[f'{px}wpr']=(hh14-c)/(hh14-ll14+1e-9)
    tp=(h+l+c)/3; tp_sma=tp.rolling(20).mean(); tp_mad=tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())),raw=True); d[f'{px}cci']=(tp-tp_sma)/(0.015*tp_mad+1e-9)/200
    tenkan=(h.rolling(9).max()+l.rolling(9).min())/2; kijun=(h.rolling(26).max()+l.rolling(26).min())/2
    senA=((tenkan+kijun)/2).shift(26); senB=((h.rolling(52).max()+l.rolling(52).min())/2).shift(26); cloud_mid=(senA+senB)/2
    d[f'{px}ichi_tk']=(tenkan-kijun)/(c+1e-9); d[f'{px}ichi_pos']=(c-cloud_mid)/(c+1e-9); d[f'{px}ichi_cld']=(senA-senB)/(c+1e-9)
    vsma2=v.rolling(20).mean(); d[f'{px}vr']=v/(vsma2+1e-9); d[f'{px}vr5']=v.rolling(5).mean()/(vsma2+1e-9)
    obv=(np.sign(c.diff())*v).fillna(0).cumsum(); obv_m=obv.rolling(30).mean(); obv_s=obv.rolling(30).std(); d[f'{px}obv']=(obv-obv_m)/(obv_s+1e-9)
    for n in [1,2,3,6,12,24]: d[f'{px}ret{n}']=c.pct_change(n)
    for n in [10,20]: d[f'{px}rvol{n}']=ret1.rolling(n).std()
    d[f'{px}skew20']=ret1.rolling(20).skew(); d[f'{px}kurt20']=ret1.rolling(20).kurt(); d[f'{px}skew10']=ret1.rolling(10).skew(); d[f'{px}autocorr']=ret1.rolling(20).corr(ret1.shift(1))
    d[f'{px}body']=(c-o)/(atr14+1e-9); d[f'{px}hl']=(h-l)/(c+1e-9); d[f'{px}upper']=(h-c.clip(lower=o))/(atr14+1e-9); d[f'{px}lower']=(c.clip(upper=o)-l)/(atr14+1e-9)
    # Anchored-VWAP distance (NEWFEAT)
    ts=pd.to_datetime(d['timestamp']); iso=ts.dt.isocalendar()
    wk=(iso.year*100+iso.week).values; mo=(ts.dt.year*100+ts.dt.month).values; pv=((h+l+c)/3*v)
    gw=pd.DataFrame({'pv':pv.values,'v':v.values,'k':wk}); avw=(gw.groupby('k')['pv'].cumsum()/(gw.groupby('k')['v'].cumsum()+1e-9)).values
    gm=pd.DataFrame({'pv':pv.values,'v':v.values,'k':mo}); avm=(gm.groupby('k')['pv'].cumsum()/(gm.groupby('k')['v'].cumsum()+1e-9)).values
    d[f'{px}vwap_dist_w']=(c.values-avw)/(atr14.values+1e-9); d[f'{px}vwap_dist_m']=(c.values-avm)/(atr14.values+1e-9)
    # Path entropy: permutation entropy (Bandt-Pompe, order d=3) on LOG-PRICE — must match trainer
    _x=np.log(c.clip(lower=1e-9)).values; _a=np.roll(_x,2); _b=np.roll(_x,1)
    code=((_a<_b).astype(np.int64)*4+(_a<_x).astype(np.int64)*2+(_b<_x).astype(np.int64)); code[:2]=-1
    oh=np.zeros((len(_x),8)); ok=code>=0; oh[np.arange(len(_x))[ok],code[ok]]=1.0
    cnt=pd.DataFrame(oh).rolling(50).sum().values; pp=cnt/(cnt.sum(1,keepdims=True)+1e-9)
    with np.errstate(divide='ignore',invalid='ignore'):
        ent=-(np.where(pp>0,pp*np.log(pp),0.0)).sum(1)/np.log(6)
    d[f'{px}pent']=ent; d[f'{px}pent_chg']=pd.Series(ent,index=d.index)-pd.Series(ent,index=d.index).shift(6)
    return d

def triple_barrier(c,h,l,ltp,lsl,stp,ssl,T):
    n=len(c); lab=np.full(n,np.nan)
    for i in range(n-1):
        ref=c[i]; ltl=ref*(1+ltp); lsll=ref*(1-lsl); stl=ref*(1-stp); ssll=ref*(1+ssl); lw=sw=None
        for j in range(i+1,min(i+1+T,n)):
            hi,lo=h[j],l[j]
            if lw is None:
                u,dn=hi>=ltl,lo<=lsll
                if u and dn: lw=False
                elif u: lw=True
                elif dn: lw=False
            if sw is None:
                dn,u=lo<=stl,hi>=ssll
                if dn and u: sw=False
                elif dn: sw=True
                elif u: sw=False
            if lw is not None and sw is not None: break
        L,S=(lw is True),(sw is True)
        if L and not S: lab[i]=1
        elif S and not L: lab[i]=0
    return lab

def ind_cols(df,prefix): return ['timestamp']+[c for c in df.columns if c.startswith(prefix)]
def build_base(files):
    d15=add_indicators(load_csv(os.path.join(DATA_DIR,files['15m'])),'15m_'); d1h=add_indicators(load_csv(os.path.join(DATA_DIR,files['1h'])),'1h_')
    d4h=add_indicators(load_csv(os.path.join(DATA_DIR,files['4h'])),'4h_'); d1d=add_indicators(load_csv(os.path.join(DATA_DIR,files['1d'])),'1d_')
    base=d4h.copy()
    for other,px in [(d1h,'1h_'),(d1d,'1d_'),(d15,'15m_')]:
        base=pd.merge_asof(base.sort_values('timestamp'),other[ind_cols(other,px)].sort_values('timestamp'),on='timestamp',direction='backward')
    base=base.sort_values('timestamp').reset_index(drop=True)
    for a,b,name in [('4h_rsi14','1d_rsi14','div_rsi_4h1d'),('4h_rsi14','1h_rsi14','div_rsi_4h1h'),('4h_macdh','1d_macdh','div_macdh_4h1d')]:
        if a in base.columns and b in base.columns: base[name]=base[a]-base[b]
    if '4h_vr' in base.columns and '1d_vr' in base.columns: base['div_vol_4h1d']=base['4h_vr']/(base['1d_vr']+1e-9)
    return base

def rolling_absorption(R,window):
    T=R.shape[0]; out=np.full(T,np.nan)
    for t in range(window,T):
        win=R[t-window:t]; m=~np.isnan(win).any(axis=0); w=win[:,m]
        if w.shape[1]<2: continue
        ev=np.linalg.eigvalsh(np.cov(w,rowvar=False)); ev=ev[ev>0]
        if ev.size: out[t]=ev.max()/ev.sum()
    return out

def add_cross_asset(bases):
    keys=list(bases.keys()); comp={}
    for a in keys:
        b=bases[a].set_index('timestamp'); c=b['Close']
        comp[a]=pd.DataFrame({'r1':c.pct_change(1),'r6':c.pct_change(6),'r24':c.pct_change(24),'atr':b['4h_atr14'],'rsi':b['4h_rsi14']})
    idx=sorted(set().union(*[comp[a].index for a in keys]))
    for a in keys: comp[a]=comp[a].reindex(idx)
    rank6=pd.concat([comp[a]['r6'] for a in keys],axis=1,keys=keys).rank(axis=1,pct=True)
    for a in keys:
        others=[k for k in keys if k!=a]
        mr1=pd.concat([comp[k]['r1'] for k in others],axis=1).mean(axis=1); mr6=pd.concat([comp[k]['r6'] for k in others],axis=1).mean(axis=1)
        mr24=pd.concat([comp[k]['r24'] for k in others],axis=1).mean(axis=1); matr=pd.concat([comp[k]['atr'] for k in others],axis=1).mean(axis=1); mrsi=pd.concat([comp[k]['rsi'] for k in others],axis=1).mean(axis=1)
        d=comp[a]; x=pd.DataFrame(index=idx)
        x[f'{X_PREFIX}relret1']=d['r1']-mr1; x[f'{X_PREFIX}relret6']=d['r6']-mr6; x[f'{X_PREFIX}relret24']=d['r24']-mr24
        x[f'{X_PREFIX}relvol']=d['atr']/(matr+1e-9); x[f'{X_PREFIX}relrsi']=d['rsi']-mrsi
        x[f'{X_PREFIX}beta']=d['r1'].rolling(60).cov(mr1)/(mr1.rolling(60).var()+1e-9); x[f'{X_PREFIX}corr']=d['r1'].rolling(60).corr(mr1); x[f'{X_PREFIX}rank6']=rank6[a]
        bx=bases[a].set_index('timestamp')
        for col in x.columns: bx[col]=x[col]
        bases[a]=bx.reset_index()
    return bases

def add_absorption(bases):
    keys=list(bases.keys()); comp={a:bases[a].set_index('timestamp')['Close'].pct_change(1) for a in keys}
    idx=sorted(set().union(*[comp[a].index for a in keys])); R=pd.concat([comp[a].reindex(idx) for a in keys],axis=1,keys=keys)
    ab=pd.Series(rolling_absorption(R.values,ABS_WINDOW),index=R.index); ab_chg=ab.diff(6)
    for a in keys:
        bx=bases[a].set_index('timestamp'); bx[f'{X_PREFIX}absorb']=ab.reindex(bx.index); bx[f'{X_PREFIX}absorb_chg']=ab_chg.reindex(bx.index); bases[a]=bx.reset_index()
    return bases


# ── model (asset-embedding transformer; matches checkpoint) ─────────
class SqueezeExcite(nn.Module):
    def __init__(s,n,r=4): super().__init__(); s.fc=nn.Sequential(nn.Linear(n,max(n//r,16)),nn.ReLU(),nn.Linear(max(n//r,16),n),nn.Sigmoid())
    def forward(s,x): return x*s.fc(x.mean(1)).unsqueeze(1)
class PatchEmbed(nn.Module):
    def __init__(s,n,patch=PATCH_SIZE,d=D_MODEL): super().__init__(); s.p=patch; s.proj=nn.Linear(n*patch,d); s.norm=nn.LayerNorm(d)
    def forward(s,x):
        B,T,Fd=x.shape; pad=(s.p-T%s.p)%s.p
        if pad: x=F.pad(x,(0,0,0,pad))
        return s.norm(s.proj(x.reshape(B,-1,s.p*Fd)))
class ConvStem(nn.Module):   # must match btc_eth_sol_cross_train.py exactly (layer names + forward)
    def __init__(s,n,patch=PATCH_SIZE,d=D_MODEL):
        super().__init__(); s.p=patch
        s.conv1=nn.Conv1d(n,d,kernel_size=3,padding=1); s.conv2=nn.Conv1d(d,d,kernel_size=3,padding=1)
        s.down=nn.Conv1d(d,d,kernel_size=patch,stride=patch); s.act=nn.GELU(); s.norm=nn.LayerNorm(d)
    def forward(s,x):
        B,T,Fd=x.shape; pad=(s.p-T%s.p)%s.p
        if pad: x=F.pad(x,(0,0,0,pad))
        x=x.transpose(1,2); x=s.act(s.conv1(x)); x=s.act(s.conv2(x))
        return s.norm(s.down(x).transpose(1,2))
class RoPEEmbedding(nn.Module):
    def __init__(s,dim,base=10000): super().__init__(); inv=1.0/(base**(torch.arange(0,dim,2).float()/dim)); s.register_buffer('inv_freq',inv)
    def forward(s,L,device): t=torch.arange(L,device=device).float(); fr=torch.outer(t,s.inv_freq); emb=torch.cat([fr,fr],-1); return emb.cos(),emb.sin()
def _rh(x): h=x.shape[-1]//2; return torch.cat([-x[...,h:],x[...,:h]],-1)
def _rope(q,k,cos,sin): cos=cos[None,None];sin=sin[None,None]; return q*cos+_rh(q)*sin,k*cos+_rh(k)*sin
def _slopes(nh):
    def sl(n): return [2**(-8*i/n) for i in range(1,n+1)]
    if (nh&(nh-1))==0: return torch.tensor(sl(nh),dtype=torch.float32)
    p=2**int(np.floor(np.log2(nh))); return torch.tensor(sl(p)+sl(2*p)[0::2][:nh-p],dtype=torch.float32)
def _alibi(nh,L,device):
    s=_slopes(nh).to(device); dist=(torch.arange(L,device=device).unsqueeze(0)-torch.arange(L,device=device).unsqueeze(1)).abs().float(); return -s.view(-1,1,1)*dist.unsqueeze(0)
class RelativeAttention(nn.Module):
    def __init__(s,d_model=D_MODEL,n_heads=N_HEADS,dropout=DROPOUT):
        super().__init__(); s.H=n_heads; s.dh=d_model//n_heads; s.scl=s.dh**-0.5
        s.qkv=nn.Linear(d_model,3*d_model,bias=False); s.out=nn.Linear(d_model,d_model,bias=False); s.drop=nn.Dropout(dropout); s.rope=RoPEEmbedding(s.dh)
    def forward(s,x):
        B,L,D=x.shape; H,dh=s.H,s.dh; Q,K,V=s.qkv(x).chunk(3,-1)
        Q=Q.view(B,L,H,dh).transpose(1,2); K=K.view(B,L,H,dh).transpose(1,2); V=V.view(B,L,H,dh).transpose(1,2)
        cos,sin=s.rope(L,x.device); Q,K=_rope(Q,K,cos,sin)
        logits=(Q@K.transpose(-2,-1))*s.scl+_alibi(H,L,x.device)
        return s.out((s.drop(logits.softmax(-1))@V).transpose(1,2).reshape(B,L,D))
class TransformerBlock(nn.Module):
    def __init__(s):
        super().__init__(); s.n1=nn.LayerNorm(D_MODEL); s.attn=RelativeAttention(); s.n2=nn.LayerNorm(D_MODEL)
        s.ff=nn.Sequential(nn.Linear(D_MODEL,D_FF),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(D_FF,D_MODEL),nn.Dropout(DROPOUT))
    def forward(s,x): x=x+s.attn(s.n1(x)); x=x+s.ff(s.n2(x)); return x
class TemporalTransformer(nn.Module):
    def __init__(s,n_feat,n_assets=3,use_emb=True,use_convstem=False):
        super().__init__(); s.se=SqueezeExcite(n_feat)
        s.embed=ConvStem(n_feat) if use_convstem else PatchEmbed(n_feat)
        s.blocks=nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        s.use_emb=use_emb
        if use_emb: s.asset_emb=nn.Embedding(n_assets,D_MODEL)
        s.head=nn.Sequential(nn.LayerNorm(D_MODEL),nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(D_MODEL//2,2))
    def forward(s,x,aid=None):
        x=s.se(x); tok=s.embed(x)
        for blk in s.blocks: tok=blk(tok)
        pooled=tok.mean(1)
        if s.use_emb and aid is not None: pooled=pooled+s.asset_emb(aid)
        return s.head(pooled)


def build():
    ckpt=torch.load(CKPT_PATH,map_location='cpu',weights_only=False)
    cfg,scaler,feat_cols=ckpt['model_cfg'],ckpt['scaler'],ckpt['feat_cols']
    seq_len=ckpt['seq_len']; vp33,vp67=ckpt.get('vol_p33'),ckpt.get('vol_p67')
    model=TemporalTransformer(cfg['n_feat'],n_assets=cfg.get('n_assets',3),use_emb=cfg.get('use_emb',True),use_convstem=cfg.get('use_convstem',False))
    model.load_state_dict(ckpt['model_state']); model.eval()
    print(f"  model n_feat={cfg['n_feat']}  use_emb={cfg.get('use_emb')}  use_convstem={cfg.get('use_convstem',False)}")

    print("  building BTC+ETH+SOL features (this includes the absorption rolling-PCA)...")
    bases={a:build_base(f) for a,f in ASSETS.items()}
    bases=add_cross_asset(bases); bases=add_absorption(bases)
    btc=bases['btc'].dropna(subset=feat_cols).reset_index(drop=True)
    X=np.clip(scaler.transform(btc[feat_cols].values.astype(np.float32)),-6,6).astype(np.float32)
    ts=btc['timestamp'].values; close=btc['Close'].values; high=btc['High'].values; low=btc['Low'].values; n=len(btc)

    probs=np.full(n,np.nan); seqs,idxs=[],[]
    for i in range(seq_len-1,n):
        seqs.append(X[i-seq_len+1:i+1]); idxs.append(i)
        if len(seqs)==512:
            xb=torch.from_numpy(np.stack(seqs)); aid=torch.zeros(len(seqs),dtype=torch.int64)
            with torch.no_grad(): probs[idxs]=torch.softmax(model(xb,aid),1)[:,1].numpy()
            seqs,idxs=[],[]
    if seqs:
        xb=torch.from_numpy(np.stack(seqs)); aid=torch.zeros(len(seqs),dtype=torch.int64)
        with torch.no_grad(): probs[idxs]=torch.softmax(model(xb,aid),1)[:,1].numpy()

    vol_idx=feat_cols.index(P.VOL_FEATURE); vol=X[:,vol_idx]
    if vp33 is None: vp33,vp67=np.percentile(vol,[33,67])
    long_thr=np.where(vol>vp67,P.THRESH_HIGH_VOL[0],np.where(vol<vp33,P.THRESH_LOW_VOL[0],P.LONG_THRESH))
    short_thr=np.where(vol>vp67,P.THRESH_HIGH_VOL[1],np.where(vol<vp33,P.THRESH_LOW_VOL[1],P.SHORT_THRESH))
    regime=np.where(vol>vp67,'high',np.where(vol<vp33,'low','mid'))
    sig=np.where(probs>=long_thr,'LONG',np.where(probs<=short_thr,'SHORT','NEUTRAL')); sig[np.isnan(probs)]='NEUTRAL'

    lab=triple_barrier(close,high,low,LONG_TP,LONG_SL,SHORT_TP,SHORT_SL,TB_TIMEOUT); valid=~np.isnan(lab)
    lab_end=np.array([i for i in range(seq_len,n+1) if valid[i-1]])
    t_tr=lab_end[int(len(lab_end)*TRAIN_FRAC)]-1; t_va=lab_end[int(len(lab_end)*VAL_FRAC)]-1
    split=np.where(np.arange(n)<t_tr,'train',np.where(np.arange(n)<t_va,'val','test'))
    return dict(ts=ts,close=close,high=high,low=low,n=n,probs=probs,sig=sig,regime=regime,split=split)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',default='test',choices=['test','val','train','full']); args=ap.parse_args()
    print(f"Model: {os.path.basename(CKPT_PATH)}  (emb + cross-asset + VWAP + Absorption, BTC asset_id=0)")
    d=build()
    span=f"{pd.Timestamp(d['ts'][0]):%Y-%m-%d} -> {pd.Timestamp(d['ts'][-1]):%Y-%m-%d}"
    print(f"  candles:{d['n']:,}  span:{span}  fees:{B.FEE_PER_SIDE*1e4:.0f}bps/side")
    print(f"  split sizes -> train:{(d['split']=='train').sum():,}  val:{(d['split']=='val').sum():,}  test:{(d['split']=='test').sum():,}")
    trades=B.run_trades(d,args.split)
    print(f"\n=== PER-SIGNAL EXPECTANCY  (split={args.split}) ===")
    B.stats('ALL',trades); B.stats('LONG',trades[trades.direction=='LONG']); B.stats('SHORT',trades[trades.direction=='SHORT'])
    for reg in ('low','mid','high'): B.stats(f'regime={reg}',trades[trades.regime==reg])
    eq,used=B.equity_curve(trades,d)
    if len(eq)>1:
        mdd=B.max_drawdown(eq['equity'].values); years=(eq['timestamp'].iloc[-1]-eq['timestamp'].iloc[0])/np.timedelta64(365,'D')
        cagr=eq['equity'].iloc[-1]**(1/max(years,1e-9))-1
        m=(d['split']==args.split) if args.split!='full' else np.ones(d['n'],bool); c=d['close'][m]; bh=c[-1]/c[0]-1
        print(f"\n=== STRATEGY EQUITY (non-overlapping, compounded) ===")
        print(f"  trades taken:{len(used)}  total return:{(eq['equity'].iloc[-1]-1)*100:+.2f}%  CAGR:{cagr*100:+.2f}%  maxDD:{mdd*100:.2f}%")
        print(f"  buy & hold  :{bh*100:+.2f}% (same window)")
        plt.figure(figsize=(11,5)); plt.plot(eq['timestamp'],eq['equity'],label='Strategy (emb_cross_nf)',color='darkgreen')
        plt.plot(pd.to_datetime(d['ts'][m]),c/c[0],label='Buy & Hold',color='gray',alpha=0.7); plt.axhline(1.0,color='k',lw=0.6,ls=':')
        plt.title(f'Newest model backtest — {args.split}  ({2*B.FEE_PER_SIDE*100:.2f}% round-trip)'); plt.ylabel('Equity (×)'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        out=os.path.join(SAVE_DIR,f'backtest_cross_nf_{args.split}.png'); plt.savefig(out,dpi=150); plt.close(); print(f"  equity plot -> {out}")
    trades.to_csv(os.path.join(SAVE_DIR,f'backtest_cross_nf_trades_{args.split}.csv'),index=False)
    print(f"\nSaved trades -> backtest_cross_nf_trades_{args.split}.csv")


if __name__ == '__main__':
    main()
