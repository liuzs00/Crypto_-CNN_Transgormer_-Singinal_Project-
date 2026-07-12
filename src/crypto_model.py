"""
crypto_model.py — Shared model definitions for the production line.

ONE source of truth for the Temporal Transformer so the trainer, backtest and any future
inference script reconstruct exactly the same network — eliminating the train/serve parity
bug class (e.g. the attention-residual bug, and hand-porting ConvStem into multiple files).

The architecture is reconstructed from `model_cfg` stored in the checkpoint:
    TemporalTransformer(n_feat, n_assets, use_emb, use_convstem)

Hyperparameters (d_model, heads, layers, …) are intrinsic to the trained weights and live
here as module constants; do not change them without retraining.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── architecture hyperparameters (must match trained checkpoints) ──
D_MODEL    = 256
N_HEADS    = 8
N_LAYERS   = 3
D_FF       = 512
DROPOUT    = 0.25
PATCH_SIZE = 4
LABEL_SMOOTH = 0.10


class SqueezeExcite(nn.Module):
    def __init__(self, n_feat, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(n_feat, max(n_feat//r, 16)), nn.ReLU(),
                                nn.Linear(max(n_feat//r, 16), n_feat), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x.mean(1)).unsqueeze(1)


class PatchEmbed(nn.Module):
    """Linear patch tokenizer (CONVSTEM=0 ablation)."""
    def __init__(self, n_feat, patch=PATCH_SIZE, d=D_MODEL):
        super().__init__(); self.p = patch; self.proj = nn.Linear(n_feat*patch, d); self.norm = nn.LayerNorm(d)
    def forward(self, x):
        B, T, Fd = x.shape; pad = (self.p - T % self.p) % self.p
        if pad: x = F.pad(x, (0, 0, 0, pad))
        return self.norm(self.proj(x.reshape(B, -1, self.p*Fd)))


class ConvStem(nn.Module):
    """Temporal CNN stem (production default): overlapping convs capture local candle motifs
    ACROSS patch boundaries (which the linear patch can't), then a strided conv downsamples
    T→T/patch tokens. Drop-in replacement for PatchEmbed; transformer handles global deps."""
    def __init__(self, n_feat, patch=PATCH_SIZE, d=D_MODEL):
        super().__init__(); self.p = patch
        self.conv1 = nn.Conv1d(n_feat, d, kernel_size=3, padding=1)      # local motifs, full temporal res
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, padding=1)           # deeper local (receptive field ~5)
        self.down  = nn.Conv1d(d, d, kernel_size=patch, stride=patch)    # → T/patch tokens
        self.act = nn.GELU(); self.norm = nn.LayerNorm(d)
    def forward(self, x):
        B, T, Fd = x.shape; pad = (self.p - T % self.p) % self.p
        if pad: x = F.pad(x, (0, 0, 0, pad))
        x = x.transpose(1, 2)                                            # (B,F,T)
        x = self.act(self.conv1(x)); x = self.act(self.conv2(x))        # (B,d,T)
        x = self.down(x).transpose(1, 2)                                # (B,T/patch,d)
        return self.norm(x)


class RoPEEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__(); inv = 1.0/(base**(torch.arange(0, dim, 2).float()/dim)); self.register_buffer('inv_freq', inv)
    def forward(self, L, device):
        t = torch.arange(L, device=device).float(); fr = torch.outer(t, self.inv_freq)
        emb = torch.cat([fr, fr], -1); return emb.cos(), emb.sin()


def _rotate_half(x): h = x.shape[-1]//2; return torch.cat([-x[..., h:], x[..., :h]], -1)
def _apply_rope(q, k, cos, sin): cos = cos[None, None]; sin = sin[None, None]; return q*cos+_rotate_half(q)*sin, k*cos+_rotate_half(k)*sin
def _alibi_slopes(nh):
    def sl(n): return [2**(-8*i/n) for i in range(1, n+1)]
    if (nh & (nh-1)) == 0: return torch.tensor(sl(nh), dtype=torch.float32)
    p = 2**int(np.floor(np.log2(nh))); return torch.tensor(sl(p)+sl(2*p)[0::2][:nh-p], dtype=torch.float32)
def _alibi_bias(nh, L, device):
    s = _alibi_slopes(nh).to(device)
    dist = (torch.arange(L, device=device).unsqueeze(0)-torch.arange(L, device=device).unsqueeze(1)).abs().float()
    return -s.view(-1, 1, 1)*dist.unsqueeze(0)


class RelativeAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__(); self.H = n_heads; self.dh = d_model//n_heads; self.scl = self.dh**-0.5
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False); self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout); self.rope = RoPEEmbedding(self.dh)
    def forward(self, x, return_attn=False):
        B, L, D = x.shape; H, dh = self.H, self.dh
        Q, K, V = self.qkv(x).chunk(3, -1)
        Q = Q.view(B, L, H, dh).transpose(1, 2); K = K.view(B, L, H, dh).transpose(1, 2); V = V.view(B, L, H, dh).transpose(1, 2)
        cos, sin = self.rope(L, x.device); Q, K = _apply_rope(Q, K, cos, sin)
        logits = (Q@K.transpose(-2, -1))*self.scl + _alibi_bias(H, L, x.device)
        attn_w = logits.softmax(-1)
        out = self.out((self.drop(attn_w)@V).transpose(1, 2).reshape(B, L, D))
        return (out, attn_w.detach()) if return_attn else out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.n1 = nn.LayerNorm(D_MODEL); self.attn = RelativeAttention(); self.n2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_FF, D_MODEL), nn.Dropout(DROPOUT))
    def forward(self, x, return_attn=False):
        if return_attn:
            ao, aw = self.attn(self.n1(x), return_attn=True); x = x+ao; x = x+self.ff(self.n2(x)); return x, aw
        x = x+self.attn(self.n1(x)); x = x+self.ff(self.n2(x)); return x


class TemporalTransformer(nn.Module):
    def __init__(self, n_feat, n_assets=3, use_emb=True, use_convstem=True):
        super().__init__(); self.se = SqueezeExcite(n_feat)
        self.embed = ConvStem(n_feat) if use_convstem else PatchEmbed(n_feat)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.use_emb = use_emb
        if use_emb: self.asset_emb = nn.Embedding(n_assets, D_MODEL)
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, D_MODEL//2), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(D_MODEL//2, 2))
    def forward(self, x, aid=None, return_attn=False):
        x = self.se(x); tok = self.embed(x)
        if return_attn:
            attns = []
            for blk in self.blocks: tok, aw = blk(tok, return_attn=True); attns.append(aw)
            pooled = tok.mean(1)
            if self.use_emb and aid is not None: pooled = pooled + self.asset_emb(aid)
            return self.head(pooled), attns
        for blk in self.blocks: tok = blk(tok)
        pooled = tok.mean(1)
        if self.use_emb and aid is not None: pooled = pooled + self.asset_emb(aid)
        return self.head(pooled)

    @classmethod
    def from_cfg(cls, cfg):
        """Reconstruct from a checkpoint's model_cfg dict (used by backtest / inference)."""
        return cls(cfg['n_feat'], n_assets=cfg.get('n_assets', 3),
                   use_emb=cfg.get('use_emb', True), use_convstem=cfg.get('use_convstem', False))


class FocalLS(nn.Module):
    """Focal loss + label smoothing."""
    def __init__(self, gamma=2.0, smooth=LABEL_SMOOTH, weight=None):
        super().__init__(); self.gamma = gamma; self.smooth = smooth; self.weight = weight
    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            y_sm = torch.full_like(logits, self.smooth/(n-1)); y_sm.scatter_(1, targets.unsqueeze(1), 1.0-self.smooth)
        log_p = F.log_softmax(logits, 1); ce = -(y_sm*log_p).sum(1)
        pt = torch.exp(-F.cross_entropy(logits, targets, weight=self.weight, reduction='none'))
        return (((1-pt)**self.gamma)*ce).mean()
