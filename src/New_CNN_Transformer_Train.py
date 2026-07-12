"""
New_CNN_Transformer_Train.py — Production model trainer (BTC+ETH+SOL pooled).

Thin orchestrator: feature construction lives in crypto_features, the network in crypto_model,
and evaluation/attention in crypto_metrics. This file only does the train-specific glue —
leak-safe split, scaler fit, the training loop, and checkpoint save — so the feature/model/
metric logic is shared verbatim with the backtest and predict scripts (no parity drift).

Env flags (all default to the production config):
    NEWFEAT=1  CROSS=1  ASSET_EMB=1  CONVSTEM=1   (set to 0 for ablations)
    SEED=<int> for a reproducible paired A/B   ·   BTC_ONLY=1 for the single-asset ablation

Outputs: btc_eth_sol_pooled_model_{TAG}.pth · sol_pooled_curves_{TAG}.png · *_signals_{TAG}.csv
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
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import crypto_features as FE
from crypto_features import (ASSET_ID, N_ASSETS, SEQ_LEN, TAG,
                             BTC_ONLY, USE_ASSET_EMB, USE_CROSS, USE_NEWFEAT, USE_CONVSTEM)
from crypto_model import TemporalTransformer, FocalLS, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, PATCH_SIZE, LABEL_SMOOTH
import crypto_metrics as M

warnings.filterwarnings('ignore')

# Optional reproducibility: fix weight-init, shuffling and dropout for a clean paired A/B.
_SEED = os.environ.get('SEED')
if _SEED is not None:
    import random as _random
    _random.seed(int(_SEED)); np.random.seed(int(_SEED)); torch.manual_seed(int(_SEED))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(_SEED))

from paths import MODELS_DIR, OUTPUTS_DIR
BATCH=64; EPOCHS=100; LR_MAX=1e-3; WARMUP_EP=20; PATIENCE=20
TRAIN_FRAC=0.70; VAL_FRAC=0.85
AUG_NOISE=0.02; AUG_MASK_P=0.10; SHORT_WEIGHT_FACTOR=1.0

# ── build features (shared pipeline) ──────────────────────────────
print(f"BTC_ONLY:{BTC_ONLY}  AssetEmb:{'ON' if USE_ASSET_EMB else 'OFF'}  "
      f"Cross:{'ON' if USE_CROSS else 'OFF'}  NewFeat:{'ON' if USE_NEWFEAT else 'OFF'}  tag={TAG}")
print("Building BTC + ETH + SOL feature bases …")
bases = FE.build_bases()
print(f"  cross-asset/absorption cols: {sum(c.startswith(FE.X_PREFIX) for c in bases['btc'].columns)}")
feat_cols = FE.select_feat_cols(bases)
for a in bases: bases[a] = bases[a].dropna(subset=feat_cols).reset_index(drop=True)
print(f"  feat_cols:{len(feat_cols)}  " + "  ".join(f"{a}:{len(bases[a]):,}" for a in bases))


def labeled_positions(base):
    X = base[feat_cols].values.astype(np.float32); lab = base['label'].values; ts = base['timestamp'].values
    valid = ~np.isnan(lab)
    pos = [(i, int(lab[i-1])) for i in range(SEQ_LEN, len(X)+1) if valid[i-1]]
    return X, ts, pos


data = {a: labeled_positions(bases[a]) for a in bases}

# BTC split boundaries (leak-safe: from BTC's labelled timeline)
Xb, tsb, posb = data['btc']
btc_end_ts = np.array([tsb[i-1] for i, _ in posb])
T_train = btc_end_ts[int(len(posb)*TRAIN_FRAC)]; T_val = btc_end_ts[int(len(posb)*VAL_FRAC)]
print(f"  T_train={pd.Timestamp(T_train):%Y-%m-%d}  T_val={pd.Timestamp(T_val):%Y-%m-%d}")

# shared scaler — fit on pooled TRAIN-era rows of the pool assets (BTC only if BTC_ONLY)
pool_assets = ['btc'] if BTC_ONLY else list(data.keys())
train_rows = []
for a in pool_assets:
    X, ts, pos = data[a]; b = int(np.searchsorted(ts, T_train)); train_rows.append(X[:b])
scaler = RobustScaler(quantile_range=(5, 95)); scaler.fit(np.vstack(train_rows))
Xs = {a: np.clip(scaler.transform(data[a][0]), -6, 6).astype(np.float32) for a in data}


def seqs_for(asset, lo=None, hi=None):
    X, ts, pos = data[asset]; Xsc = Xs[asset]; aid = ASSET_ID[asset]
    Xo, yo, to, ao = [], [], [], []
    for i, lbl in pos:
        e = ts[i-1]
        if (lo is None or e >= lo) and (hi is None or e < hi):
            Xo.append(Xsc[i-SEQ_LEN:i]); yo.append(lbl); to.append(e); ao.append(aid)
    if not Xo:
        z = np.empty((0, SEQ_LEN, len(feat_cols)), np.float32)
        return z, np.empty((0,), np.int64), np.empty((0,), 'datetime64[ns]'), np.empty((0,), np.int64)
    return np.stack(Xo).astype(np.float32), np.array(yo, np.int64), np.array(to), np.array(ao, np.int64)


# Train = pooled assets before T_train (BTC only if BTC_ONLY); Val/Test = BTC only
trX, trY, trA = [], [], []
for a in pool_assets:
    X, y, _, A = seqs_for(a, hi=T_train); trX.append(X); trY.append(y); trA.append(A)
Xtr = np.concatenate(trX); ytr = np.concatenate(trY); atr_id = np.concatenate(trA)
Xva, yva, _, _     = seqs_for('btc', lo=T_train, hi=T_val)
Xte, yte, ts_te, _ = seqs_for('btc', lo=T_val)
vol_idx = feat_cols.index(M.VOL_FEATURE)
atr_tr_btc = Xtr[atr_id == ASSET_ID['btc']][:, -1, vol_idx]
counts = Counter(ytr.tolist())
per_asset = {a: int((atr_id == ASSET_ID[a]).sum()) for a in ASSET_ID}
print(f"  train pooled:{len(ytr):,}  ({per_asset})   Short:{counts[0]:,} Long:{counts[1]:,}")
print(f"  val(btc):{len(yva):,}  test(btc):{len(yte):,}")


class SeqDS(Dataset):
    def __init__(self, X, y, A, augment=False):
        self.X = torch.from_numpy(X.astype(np.float32)); self.y = torch.from_numpy(y.astype(np.int64))
        self.A = torch.from_numpy(A.astype(np.int64)); self.augment = augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            x = x + AUG_NOISE*torch.randn_like(x)
            m = torch.bernoulli(torch.full((x.shape[-1],), AUG_MASK_P)).bool(); x[:, m] = 0.0
        return x, self.y[i], self.A[i]


train_dl = DataLoader(SeqDS(Xtr, ytr, atr_id, augment=True), batch_size=BATCH, shuffle=True, drop_last=True)
val_dl   = DataLoader(SeqDS(Xva, yva, np.zeros(len(yva), np.int64)), batch_size=BATCH, shuffle=False)
test_dl  = DataLoader(SeqDS(Xte, yte, np.zeros(len(yte), np.int64)), batch_size=BATCH, shuffle=False)

# ── model + training ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
n_feat = Xtr.shape[2]
model = TemporalTransformer(n_feat, n_assets=N_ASSETS, use_emb=USE_ASSET_EMB, use_convstem=USE_CONVSTEM).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}  (asset_emb={'on' if USE_ASSET_EMB else 'off'})")
total = sum(counts.values())
w = torch.tensor([total/(2*counts[0])*SHORT_WEIGHT_FACTOR, total/(2*counts[1])], dtype=torch.float32).to(device)
criterion = FocalLS(gamma=2.0, smooth=LABEL_SMOOTH, weight=w)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-3)
def lr_lambda(ep):
    if ep < WARMUP_EP: return (ep+1)/WARMUP_EP
    p = (ep-WARMUP_EP)/max(1, EPOCHS-WARMUP_EP); return 0.5*(1+np.cos(np.pi*p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

best_val, best_state, no_imp = float('inf'), None, 0
hist = {'tl': [], 'vl': [], 'va': []}
print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}")
print("-"*48)
for ep in range(1, EPOCHS+1):
    model.train(); tl = 0.0
    for xb, yb, ab in train_dl:
        xb, yb, ab = xb.to(device), yb.to(device), ab.to(device)
        optimizer.zero_grad(); loss = criterion(model(xb, ab), yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); tl += loss.item()
    tl /= len(train_dl); scheduler.step()
    model.eval(); vl, vp, vt = 0.0, [], []
    with torch.no_grad():
        for xb, yb, ab in val_dl:
            xb, yb, ab = xb.to(device), yb.to(device), ab.to(device); out = model(xb, ab)
            vl += criterion(out, yb).item(); vp.extend(out.argmax(1).cpu().tolist()); vt.extend(yb.cpu().tolist())
    vl /= len(val_dl); va = accuracy_score(vt, vp)
    hist['tl'].append(tl); hist['vl'].append(vl); hist['va'].append(va)
    if vl < best_val:
        best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0; mk = ' *'
    else: no_imp += 1; mk = f'  (no improve {no_imp}/{PATIENCE})'
    print(f"{ep:4d}  {tl:8.4f}  {vl:8.4f}  {va:7.4f}  {optimizer.param_groups[0]['lr']:9.2e}{mk}")
    if no_imp >= PATIENCE: print(f"  → Early stop at epoch {ep}"); break
print(f"\nBest val loss (BTC): {best_val:.4f}")

# ── evaluation (shared, incl. extended metrics A–E) + save ────────
model.load_state_dict(best_state)
_btc = bases['btc']
_price = (_btc['Close'].values, _btc['High'].values, _btc['Low'].values,
          np.searchsorted(_btc['timestamp'].values, ts_te), FE.TB_TIMEOUT)   # MAE path
res = M.evaluate(model, test_dl, Xte, vol_idx, atr_tr_btc, device, TAG,
                 feat_cols=feat_cols, X_train=Xtr, price_path=_price)         # +PSI (Xtr) +MAE
sig, regime, p_long, tprobs, ttrue, p33, p67 = (res['sig'], res['regime'], res['p_long'],
                                                res['tprobs'], res['ttrue'], res['p33'], res['p67'])

ckpt = os.path.join(MODELS_DIR, f'btc_eth_sol_pooled_model_{TAG}.pth')
torch.save({'model_state': best_state,
            'model_cfg': dict(n_feat=n_feat, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                              drop=DROPOUT, patch_size=PATCH_SIZE, n_assets=N_ASSETS,
                              use_emb=USE_ASSET_EMB, use_convstem=USE_CONVSTEM),
            'scaler': scaler, 'feat_cols': feat_cols, 'seq_len': SEQ_LEN, 'history': hist,
            'vol_p33': p33, 'vol_p67': p67}, ckpt)
print(f"\nCheckpoint → {ckpt}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(hist['tl'], label='Train(pooled)'); ax[0].plot(hist['vl'], label='Val(BTC)')
ax[0].set_title(f'BTC+ETH+SOL ({TAG}) Loss'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(hist['va'], color='green'); ax[1].axhline(0.5, color='gray', ls='--'); ax[1].set_title('Val Acc (BTC)'); ax[1].grid(True)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUTS_DIR, f'sol_pooled_curves_{TAG}.png'), dpi=150); plt.close()

sig_df = pd.DataFrame({'timestamp': ts_te[:len(ttrue)], 'true_label': ttrue, 'prob_short': tprobs[:, 0],
                       'prob_long': p_long, 'vol_regime': regime,
                       'signal': np.where(sig == 1, 'LONG', np.where(sig == 0, 'SHORT', 'NEUTRAL'))})
sig_df.to_csv(os.path.join(OUTPUTS_DIR, f'btc_eth_sol_pooled_signals_{TAG}.csv'), index=False)
print(f"Signals → btc_eth_sol_pooled_signals_{TAG}.csv  ({(sig_df.signal != 'NEUTRAL').sum():,} active)")

M.attention_analysis(model, test_dl, Xte, feat_cols, sig, ttrue, device, TAG, OUTPUTS_DIR,
                     SEQ_LEN, PATCH_SIZE, N_LAYERS)
