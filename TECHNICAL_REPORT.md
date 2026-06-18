# Technical Report — BTC Multi-Timeframe Temporal Transformer

**Scope.** This report documents (1) the network architecture, (2) the feature-construction
pipeline, and (3) the training, labelling, calibration, and evaluation techniques used in the
BTC long/short signal engine. All notation matches the implementation in
`btc_lstm_train.py` and `btc_predict.py`.

---

## 1. Task Definition

For each 4-hour candle *t* the model emits a probability distribution over two classes,
**Long** (1) and **Short** (0). Labels come from the **Triple-Barrier Method**; signals are
produced by passing class probabilities through a **volatility-regime-adaptive threshold**.

- Input: a length-**64** window of 4h candles (≈ 11 calendar days), each candle described by
  **256 cross-timeframe features**.
- Output: `P(Long), P(Short)` → discretised to `LONG / SHORT / NEUTRAL`.

---

## 2. Data & Labelling

### 2.1 Multi-timeframe alignment
Four raw OHLCV streams (15m, 1h, 4h, 1d) are each enriched with indicators, then aligned to
the **4h grid** using a backward `merge_asof`. This guarantees **no look-ahead**: a 4h candle
at time *t* only ever sees the most recent *completed* 1h/15m/1d values at or before *t*.

### 2.2 Triple-Barrier labelling (López de Prado)
For entry price `c = Close[t]`, define an upper barrier `c(1+0.015)`, a lower barrier
`c(1−0.03)`, and a vertical (time) barrier of 4 candles:

```
for j in (t+1 … t+4):
    if High[j] ≥ TP and Low[j] ≤ SL :  break        # ambiguous → NaN
    elif High[j] ≥ TP              :  label = 1      # Long  (TP first)
    elif Low[j]  ≤ SL              :  label = 0      # Short (SL first)
# no touch within 4 candles → NaN (timeout)
```

- **Asymmetric barriers** (TP = 1.5%, SL = 3.0%) embed a defined risk/reward target into the
  label itself rather than predicting raw direction.
- **NaN rows** (timeout or same-candle ambiguity) are **excluded from the loss** but **kept in
  the feature matrix**, so the temporal context of a sequence is continuous and identical at
  train and inference time (see §5.3).
- Resulting prior is imbalanced (~73% Long / 27% Short) — a direct consequence of the
  asymmetric barriers in a multi-year uptrend, handled explicitly in §4.

---

## 3. Feature Construction (256 features)

**Design principle — scale invariance.** BTC ranges over 10×+ in price across 2018–2026.
Every feature is a **ratio, return, z-score, or ATR-normalised quantity**, so the input
distribution is stationary in scale and the same model transfers across price regimes.
Features are computed **per timeframe** (prefixes `15m_ 1h_ 4h_ 1d_`) and concatenated.

### 3.1 Market microstructure / order flow
Derived from raw exchange fields (taker-buy volume, trade count, quote volume):

| Feature | Definition | Signal |
|---|---|---|
| Buy ratio | `taker_buy / volume` ∈ [0,1] | aggressor buy pressure |
| Order-Flow Imbalance (OFI) | `Σ(buy−sell) / Σvol` over 5/10/20 | net signed flow |
| OFI momentum | `OFI₅ − OFI₂₀` | acceleration of flow |
| VPIN proxy | `Σ\|buy−sell\| / Σvol` over 5/20 | flow toxicity / informed trading |
| OFI–price divergence | rolling mean of `sign(ret) − sign(OFI)` | price up but sellers dominating → bearish |
| Amihud illiquidity | `\|ret\| / quote_vol`, normalised | price impact per dollar traded |
| Trade intensity / size | trades and vol/trade vs 20-MA | retail vs institutional footprint |

### 3.2 Volatility & jump structure
- **ATR** (7, 14) via Wilder EMA, expressed as `ATR/Close` and `ATR₇/ATR₁₄`.
- **Realised volatility** (10, 20), **Bollinger** bandwidth & %B.
- **Realised Bipower Variation jump ratio**: `RV / RBV` where
  `RV = Σrₜ²` and `RBV = (π/2)·Σ|rₜ||rₜ₋₁|`. RBV is jump-robust, so `RV/RBV > 1` flags a
  discontinuous jump (news, liquidation cascade) vs smooth diffusion.
- **Liquidity-sweep detection**: `wick × volume-spike` (a stop-hunt signature), range
  efficiency `|ret| / ((H−L)/C)`.

### 3.3 Trend, momentum, oscillators
EMA-ratio cascade (9/21/50/200), ATR-normalised MACD, RSI (7/14) scaled to [0,1],
Stochastic %K/%D, Williams %R, CCI, full **Ichimoku cloud** (TK cross, price-vs-cloud,
cloud thickness), OBV z-score, multi-horizon returns (1–24), rolling **skew/kurtosis** of
returns (distribution shape), lag-1 **return autocorrelation** (momentum vs mean-reversion),
candle-structure ratios (body, wicks, high-low).

### 3.4 Cross-timeframe divergence
Explicit interaction features capturing regime conflict between fast and slow horizons:
`4h_rsi14 − 1d_rsi14`, `4h_rsi14 − 1h_rsi14`, `4h_macdh − 1d_macdh`, `4h_vr / 1d_vr`.

### 3.5 Feature selection & cleaning
Constant columns (`nunique ≤ 1`) are dropped; rows with any NaN feature (indicator warm-up)
are removed. The surviving **256** columns are frozen and serialised in the checkpoint so
train and inference use an identical, ordered feature set.

---

## 4. Network Architecture

```
x ∈ ℝ^(B×64×256)
   │
   ▼  Squeeze-and-Excitation   (channel gating)
   ▼  Patch Embedding          (4 steps → 1 token, 64→16 tokens, →ℝ^256)
   ▼  3 × Transformer block    (pre-LN, 8-head RoPE+ALiBi attention, GELU FFN d_ff=512)
   ▼  Mean-pool over 16 tokens
   ▼  MLP head: LN → 256→128 → GELU → Dropout → 128→2
logits ∈ ℝ^(B×2)
```

≈ **1.9M trainable parameters**; `d_model=256`, `heads=8`, `layers=3`, `dropout=0.25`.

### 4.1 Squeeze-and-Excitation (SE) channel gating
`w = σ(W₂·ReLU(W₁·mean_t(x)))`, output `x ⊙ w`. A global average over time produces a
per-feature gate, letting the network **down-weight noisy indicators per sample** before any
mixing — a learned, input-conditioned feature importance.

### 4.2 Patch embedding
Groups **4 consecutive timesteps** into one token (64 → 16 tokens) and linearly projects the
`4×256` block to `d_model`, followed by LayerNorm. This **denoises high-frequency jitter**,
shortens the attention sequence (16² vs 64² cost), and mirrors the patch tokenisation idea
from vision/time-series transformers (ViT / PatchTST).

### 4.3 Attention with RoPE + ALiBi (parameter-free positional encoding)
Standard scaled dot-product attention, augmented with **two complementary** positional
mechanisms, neither of which adds parameters:

- **RoPE (Rotary Position Embedding)** — rotates Q and K by position-dependent angles so the
  dot-product `qᵢ·kⱼ` depends only on the **relative offset** `i−j`. Encodes *where* tokens
  are relative to each other.
- **ALiBi (Attention with Linear Biases)** — adds a per-head linear penalty `−mₕ·|i−j|` to the
  attention logits, a built-in **recency bias**. Encodes *how strongly distance matters*.

Both are well-suited to **non-stationary financial series** where recent context dominates
and absolute calendar position is meaningless. Pre-LayerNorm residual blocks are used for
training stability.

> **Implementation note (verified).** The residual structure is
> `x = x + attn(LN(x)); x = x + ffn(LN(x))`. A subtle bug where the attention residual was
> collapsed into the FFN residual was identified and fixed; train/inference outputs now match
> to < 1e-7 for identical inputs.

### 4.4 Head
Tokens are **mean-pooled** (order-invariant aggregation, robust to the patch count) and passed
through a 2-layer MLP to 2 logits.

---

## 5. Training Methodology

### 5.1 Loss — Focal loss + label smoothing
`FL = (1−pₜ)^γ · CE_smooth`, with `γ = 2.0` and smoothing `ε = 0.10`.

- **Focal term** `(1−pₜ)^γ` down-weights easy examples and concentrates gradient on hard,
  ambiguous candles — important when the majority Long class is easy.
- **Label smoothing** prevents overconfident logits and the associated plateau, improving
  calibration of the probabilities that the threshold layer later relies on.

### 5.2 Class imbalance
Inverse-frequency class weights `wₖ = N / (2·Nₖ)`, with an additional tunable
`SHORT_WEIGHT_FACTOR` multiplier on the minority Short class, passed into the focal CE.

### 5.3 Leakage-safe sequence construction (key correctness property)
Sequences are built from the **full, continuous row matrix** (including unlabeled timeout
rows); only sequences whose **final** row carries a valid label contribute to the loss. This
guarantees a training window and an inference window ending on the same candle contain the
**same 64 physical candles**. (An earlier version dropped unlabeled rows *before* windowing,
so training windows silently spanned ~20 days vs inference's ~11 — a temporal-context mismatch
that was diagnosed and removed.)

### 5.4 Normalisation
`RobustScaler` (5–95 quantile range) **fit on the training partition only**, transform clipped
to ±6. Median/IQR centering is robust to crypto's fat tails and outliers; the fitted scaler is
serialised in the checkpoint so inference never re-fits.

### 5.5 Optimisation schedule
- **AdamW**, `lr_max = 1e-3`, weight decay `1e-3`, gradient-norm clip `1.0`.
- **LR warmup (20 ep) → cosine decay** over 100 epochs.
- **Early stopping** on validation loss, patience 20; best-val weights restored before test.

### 5.6 Data augmentation
Applied to the training stream only: additive **Gaussian jitter** (σ = 0.02) and random
**feature masking** (p = 0.10, zero out channels). Both regularise against the noisy,
non-stationary input and discourage reliance on any single indicator.

### 5.7 Chronological split
70 / 15 / 15 **time-ordered** train/val/test (no shuffling across time). All percentile
boundaries, scaler statistics, and class weights are computed on train only.

---

## 6. Inference & Calibration

### 6.1 Volatility-regime-adaptive thresholding
The scaled `4h_atr14` at the final timestep classifies each candle into **low / mid / high**
volatility via the **training-set** p33/p67 percentiles (serialised in the checkpoint so the
regime split is identical at serve time). Each regime applies its own confidence gate:

| Regime | Long thr | Short thr |
|---|:---:|:---:|
| low | 0.55 | 0.45 |
| mid | 0.60 | 0.40 |
| high | 0.72 | 0.28 |

`p_long ≥ long_thr → LONG`, `p_long ≤ short_thr → SHORT`, else `NEUTRAL`.

### 6.2 Threshold calibration (`btc_threshold_tuning.py`)
A 3-stage search optimises the gates on held-out data:
1. **Regime boundaries** — grid over `p_low × p_high`.
2. **Per-regime thresholds** — 9×9 long/short grid per regime, under minimum-coverage and
   minimum-signal constraints.
3. **Joint fine-tune + test validation** — neighbourhood search around stage-2 winners.

Primary objective: precision weighted 40% Long / 60% Short.
Secondary: **expected PnL** proxy `prec·TP − (1−prec)·SL` per signal.

### 6.3 Expected-value framing
Break-even precision is dictated by the payoff geometry, `SL/(TP+SL)` in the entry direction.
With TP 1.5% / SL 3.0%: longs need **66.7%**, shorts (natural down-move payoff) need **33.3%**.
This is why the minority Short class can be tradeable despite lower absolute precision.

---

## 7. Evaluation & Interpretability

- **Metrics**: accuracy, per-class precision/recall/F1, confusion matrix, coverage (% filtered
  to NEUTRAL), and per-regime accuracy — reported on both the argmax baseline and the
  confidence-filtered signals.
- **Probability health check**: mean/std/min/max of `P(Long)` — a wide spread confirms the
  model is confident and calibrated rather than collapsed near 0.5.
- **Attention-map analysis**: "received attention" per patch (column-sum of the attention
  matrix), per-layer entropy (focus vs diffusion), recency ratio (late vs early patches), and
  Long-vs-Short / correct-vs-wrong attention profiles. Used to verify the model attends to
  recent context (consistent with the ALiBi prior) and to inspect failure modes.

---

## 8. Reproducibility & Serving

- Single checkpoint bundles **weights + fitted scaler + ordered feature list + regime
  boundaries + config**, so `btc_predict.py` reconstructs the exact train-time transform.
- `cmc_api.py` refreshes Binance klines, drops the still-forming candle, and **merges into the
  historical CSVs with de-duplication and continuity validation** across all four timeframes.
- Inference CLI supports a default 24h window or any explicit `--from/--to` range for
  backtesting.

---

## 9. Techniques Checklist (for quick reference)

**Architecture:** Temporal Transformer · Squeeze-and-Excitation · patch tokenisation ·
RoPE · ALiBi · pre-LayerNorm residuals · mean-pool classification head.

**Learning:** focal loss · label smoothing · inverse-frequency class weighting · AdamW ·
warmup + cosine LR · gradient clipping · early stopping · jitter & feature-mask augmentation.

**Finance / data:** triple-barrier labelling · asymmetric risk/reward barriers · ATR / Wilder
EMA · order-flow imbalance · VPIN · Amihud illiquidity · realised bipower variation (jump
detection) · Ichimoku · multi-timeframe fusion via merge-asof · scale-invariant feature design.

**Validation / serving:** chronological split · leakage-safe windowing · RobustScaler fit on
train only · volatility-regime-adaptive thresholds · expected-PnL calibration · attention-map
interpretability · deterministic train/serve parity.
