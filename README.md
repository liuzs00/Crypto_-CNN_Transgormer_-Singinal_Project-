# Crypto Multi-Asset Temporal-CNN Transformer — Long/Short Signal Engine

A research-grade deep-learning pipeline that forecasts directional trading signals for
**BTC** over a 4-hour horizon, trained on a **pooled BTC + ETH + SOL** universe. The system
fuses **four timeframes** (15m / 1h / 4h / 1d), engineers **282 scale-invariant features**
(market-microstructure, cross-asset, systemic-risk and information-theoretic), and trains a
**temporal-CNN → Transformer** (a conv tokenizer feeding RoPE + ALiBi attention) under a
**triple-barrier labelling scheme** from quantitative finance.

> The goal is not "predict the price" — it is to frame trading as a **probabilistic,
> risk-aware classification problem** and to engineer the full stack end-to-end:
> data ingestion → feature engineering → labelling → model → calibration → backtest → serve.

The repository carries two models: a single-asset **baseline** (`Baseline_Transformer_Train.py`)
and the **production multi-asset model** `emb_cross_nf` (`New_CNN_Transformer_Train.py`), which
is the focus below. Every feature and architecture change to the production model was adopted
only after a **seed-controlled A/B protocol** (§7) — most candidate ideas were *rejected*.

---

## 1. Problem Formulation — Triple-Barrier Labelling

Naïve price-direction prediction (`up` vs `down` next candle) ignores **risk/reward** and
**path dependency**. Instead, each candle is labelled with the **Triple-Barrier Method**
(López de Prado), with **independent long and short barriers**:

| Label | Meaning | Condition (≤ 4 candles ahead) |
|:---:|---|---|
| **1** (Long) | long side wins | High reaches **+3%** before Low reaches **−1.5%**, and the short side does *not* win |
| **0** (Short) | short side wins | Low reaches **−3%** before High reaches **+1.5%**, and the long side does *not* win |
| **NaN** | ambiguous / timeout | otherwise → **excluded from the loss** (but kept in the sequence for context) |

Each side carries a **2:1 reward:risk** target baked directly into the learning objective.
Excluding ambiguous rows *from the loss but not the feature matrix* keeps the temporal context
identical at train and inference time — a subtle leakage trap that this pipeline avoids.

---

## 2. Architecture — Temporal CNN → Transformer

```
   BTC + ETH + SOL  ·  4 timeframes (15m/1h/4h/1d)  →  282 features
   aligned to the 4h grid via backward merge-asof
                         │  sequence of 64 candles (≈11 days)
                         ▼
   Squeeze-and-Excitation gate        per-sample feature-channel recalibration
                         ▼
   Temporal Conv Stem                 two overlapping Conv1d(k=3) layers capture local
   (ConvStem)                         candle motifs ACROSS patch boundaries, then a strided
                         │             conv downsamples 64 → 16 tokens (learned tokenizer)
                         ▼
   ┌──────── Transformer × 3 layers ────────┐
   │  Pre-LayerNorm · 8-head attention      │
   │  RoPE  (relative position via Q/K)     │
   │  ALiBi (linear recency bias, no params)│
   │  GELU feed-forward (d_ff = 512)        │
   └─────────────────────────────────────────┘
                         ▼
   mean-pool  +  learnable per-asset embedding  →  2-layer MLP head  →  P(Long), P(Short)
```

- **~2.3M parameters**, `d_model = 256`, sequence length 64, patch size 4.
- **Temporal conv stem** replaces the usual linear patch projection. Its overlapping
  receptive field sees local candle formations *across* patch boundaries (a non-overlapping
  linear patch cannot), which is a better *inductive bias* — validation loss **improved**
  despite the extra parameters. It is the first architecture change in the project to beat the
  data/feature levers (§7). The linear-patch tokenizer remains available as an ablation.
- **Asset embedding:** a learnable per-asset bias added to the pooled vector, so shared layers
  learn universal structure while the embedding absorbs per-asset idiosyncrasy.
- **RoPE + ALiBi** combine relative position with an explicit recency prior — both
  parameter-free and well-suited to non-stationary financial series.

---

## 3. Feature Engineering (282 features)

All features are **scale-invariant** (ratios, returns, z-scores, ATR-normalised) and strictly
**causal**, so the model generalises across BTC's 10×+ price range and pools cleanly across
assets. Full dictionary in [FEATURES.md](FEATURES.md).

**Per-timeframe** (×4 timeframes)
- **Microstructure / order flow:** taker buy-pressure, signed **Order-Flow Imbalance (OFI)**,
  **VPIN** toxicity, OFI–price divergence, **Amihud illiquidity**, trade intensity/size.
- **Volatility & jumps:** ATR ratios, Bollinger %B/bandwidth, realised vol, **Realised Bipower
  Variation jump ratio** (diffusion vs jump), liquidity-sweep detection.
- **Trend / momentum:** EMA cascade (9/21/50/200), ATR-normalised MACD, RSI, Stochastic,
  Williams %R, CCI, full Ichimoku cloud, OBV, multi-horizon returns, skew/kurtosis, autocorr.
- **Anchored VWAP distance** — price vs the weekly/monthly volume-weighted cost basis.
- **Path entropy** — Bandt-Pompe **permutation entropy** of recent log-price: low = trending
  (momentum reliable), high = chop (favour reversion). Encodes path *predictability* that
  realised vol (a magnitude measure) misses.

**Cross-asset & systemic** (each asset vs the rest of the universe)
- Relative return/vol/RSI, β and correlation to the universe, cross-sectional rank.
- **Absorption Ratio** — variance share of the 1st principal component of the BTC/ETH/SOL
  return covariance (rolling PCA): a systemic-fragility / "assets in lockstep" signal.

---

## 4. Multi-Asset Pooling (leak-safe)

Training rows are **pooled across BTC + ETH + SOL** — more, more-diverse, higher-vol data —
while validation and test stay **BTC-only**:

- Train/val boundaries are taken from BTC's labelled timeline at fixed 70/85 fractions, and
  every ETH/SOL row from the val/test era is dropped → the BTC test candles are *identical* to
  the single-asset baseline, keeping every experiment comparable.
- The `RobustScaler` (5–95 quantile, clipped ±6) is fit on the **pooled training partition only**.
- Cross-asset / absorption features use only *contemporaneous* (≤ t) values of the other
  assets — no look-ahead.

---

## 5. Training Methodology

| Component | Choice | Rationale |
|---|---|---|
| Loss | **Focal loss + label smoothing (0.1)** | focus on hard examples; avoid overconfident plateau |
| Class weighting | inverse-frequency (configurable Short factor) | counter the Long/Short imbalance |
| Augmentation | Gaussian jitter + random feature masking | regularise against noisy, non-stationary inputs |
| Schedule | LR warmup → cosine decay (AdamW, grad-clip) | stable early training, smooth convergence |
| Split | **chronological** 70 / 15 / 15 | no look-ahead leakage; scaler fit on train only |
| Reproducibility | optional `SEED` (fixes init/shuffle/dropout) | clean **paired A/B** comparisons |
| Early stopping | patience on validation loss | restore best-val weights before test |

---

## 6. Volatility-Regime-Adaptive Thresholding

A single probability cutoff is suboptimal across volatility regimes. The engine classifies each
candle into **low / mid / high** ATR regimes (percentile boundaries fit on the training set,
serialised into the checkpoint) and applies regime-specific confidence gates:

| Regime | Long thr | Short thr | Intuition |
|---|:---:|:---:|---|
| Low vol | 0.55 | 0.45 | calmer tape → looser gate |
| Mid vol | 0.60 | 0.40 | balanced |
| High vol | 0.72 | 0.28 | demand high conviction in chaos |

`btc_threshold_tuning.py` performs a **3-stage grid search** (regime boundaries → per-regime
thresholds → joint fine-tune) optimising a precision-weighted + **expected-PnL** objective.

---

## 7. Empirical Adoption Protocol (how features/architecture earn their place)

Every proposed change is run as a **seed-paired A/B** (a fixed `SEED` makes weight-init,
shuffling and dropout identical, so only the change differs) and judged on **active-signal
accuracy across ≥ 3 seeds**, with a **coverage check** (an accuracy gain from simply trading
fewer bars is conservatism, not quality). Architecture changes additionally require a
**multi-seed backtest** confirmation — trusting profit factor / Calmar (robust) over total
return (compounded, path-dependent).

| Idea | Type | Verdict |
|---|---|---|
| Path entropy | feature | **adopted** — +1.3pp active acc, 3/3 seeds |
| Temporal conv stem | architecture | **adopted** — +2.66pp acc 3/3; PF 5/5, Calmar +60% in backtest |
| Wavelet (DWT) coeffs · Hawkes self-excitation · fractional differencing · gated attention pooling | — | **rejected** — did not replicate across seeds |

The recurring lesson: **data (multi-asset) and orthogonal features move the needle; heavy
architecture changes do not — except a minimal tokenizer upgrade that preserves the working
transformer.** Rejecting four plausible ideas is as much the point as adopting two.

---

## 8. Representative Results (held-out BTC test set)

| Metric | Value |
|---|---|
| Directional accuracy (argmax) | **~0.70** |
| Confidence-filtered (active-signal) accuracy | **~0.74** |
| Backtest profit factor (per-signal) | **~1.95** |
| Backtest max drawdown | **~ −24%** |

In a research backtest (test split, **0.05% fee/side, no slippage/funding, single window,
intrabar-fill assumption**) the strategy compounds strongly positive vs a negative buy-&-hold
over the same window, with both long and short sides positive-expectancy. Headline compounded
return is high but is the *least* reliable statistic; profit factor, per-trade expectancy and
drawdown (the robust measures) are what the conv-stem A/B improved.

> *Disclaimer: research project for skill demonstration. Not financial advice, not a live
> trading system. Backtests overstate live performance.*

---

## 9. Meta-Labeling — Confidence-Scaled Position Sizing

A second model decides *how much* to trade, separating **direction** from **sizing** (the
López de Prado meta-labeling pattern):

- **M1** — the frozen conv-stem transformer above — decides **direction**.
- **M2** — a **GBM + RandomForest ensemble** — scores whether each M1 signal will be
  **profitable** and emits a **confidence** + a continuous **recommended size** (0 = skip →
  1 = full position), scaling exposure by conviction and suppressing weak signals in
  unfavourable regimes.

M2 reads M1's conviction plus a curated regime/volatility context (ATR, realised vol, PCA
absorption, relative vol, ~11-day macro-vol) — it *judges the regime*, it doesn't relearn M1's
call. Training and serving share one `m2_features()`, so they can't drift.

**Validated the honest way — expanding-window walk-forward** (the only test that answers "does
the edge survive regime drift?"): train M2 on each window's past, test on its future,
6 folds × 5 seeds. **M2 lifts per-trade expectancy in 6 / 6 folds (+0.60% → +1.29%).** A
hyper-parameter search confirmed the default near-optimal and M2 well-calibrated (ECE ≈ 0.02);
the remaining lever is **retraining cadence** — the most-recent fold shows the edge decaying as
the market drifts from the training regime, so the model is retrained on a schedule.

Live signals therefore carry not just a direction but a size, e.g.
`LONG · P=0.81 · M2conf=0.67 · size 0.83×`.

---

## 10. Code Architecture & Interpretability

The production line is **modular**, with one source of truth each for features, model and
metrics — so the trainer, backtest and predict scripts cannot drift out of train/serve parity
(the bug class that previously caused a train/inference mismatch):

```
crypto_features.py   feature pipeline + config        ← imported by train / backtest / predict
crypto_model.py      ConvStem + Transformer (from_cfg) ← single model definition
crypto_metrics.py    evaluate · regime gates · attention analysis · extended metrics (A–E)
btc_meta_label.py    M2 trainer + serving lib (shared m2_features → no train/serve drift)
```

- **Extended diagnostics (A–E)** in `crypto_metrics.py`, printed every training run: calibration
  (Brier/ECE), per-class profit factor, signal autocorrelation, **PSI feature-drift**, and
  maximum adverse excursion — surfacing calibration, regime drift, and intra-trade risk.
- **Attention-map analysis** (in `crypto_metrics.py`): received-attention per patch, per-layer
  entropy, recency ratios, and Long-vs-Short / correct-vs-wrong profiles.
- **Reproducible inference:** the checkpoint bundles weights + fitted scaler + ordered
  feature list + regime boundaries + `model_cfg`, so the model is reconstructed exactly at
  serve time (`TemporalTransformer.from_cfg`); backtest output is byte-identical to the trainer.
- **Automated data layer** (`crypto_data.py`): pulls fresh Binance klines (UTC), drops the
  still-forming candle, and **merges into the historical CSVs with de-duplication + continuity
  checks** across all three assets × four timeframes.

---

## 11. Repository Structure

```
├── crypto_data.py                 # Binance fetch/update (BTC/ETH/SOL): de-dup + continuity merge
├── crypto_features.py             # SHARED feature pipeline + config (282 features)
├── crypto_model.py                # SHARED model: temporal conv stem + RoPE/ALiBi transformer
├── crypto_metrics.py              # SHARED evaluation + regime gates + attention + metrics A–E
├── New_CNN_Transformer_Train.py   # Production M1 trainer (BTC+ETH+SOL pooled) — thin orchestrator
├── Baseline_Transformer_Train.py  # Single-asset baseline trainer (frozen reference)
├── btc_backtest_cross_nf.py       # Backtest the production model (triple-barrier trade sim)
├── btc_predict_cross_nf.py        # Live signal: auto-fills the data gap → M1 dir + M2 size
├── btc_meta_label.py              # M2 meta-label (GBM+RF ensemble): confidence + sizing
├── walk_forward_m2.py             # M2 drift validation (expanding-window walk-forward)
├── m2_hp_search.py                # M2 HP / calibration / sizing-curve search
├── btc_threshold_tuning.py        # 3-stage regime/threshold calibration
├── FEATURES.md  ·  TECHNICAL_REPORT.md          # Feature dictionary · architecture deep-dive
├── DATA/                          # Multi-asset, multi-timeframe OHLCV (2018 → present, UTC)
├── btc_eth_sol_pooled_model_emb_cross_nf.pth    # Production M1 checkpoint
└── btc_eth_sol_meta_m2.pkl                       # Production M2 ensemble
```

---

## 12. Usage

```bash
# 1. Refresh data (merge latest Binance candles into the historical CSVs; de-dup + continuity)
python crypto_data.py                      # update all assets / timeframes

# 2. Train the production M1 (defaults to the production config: 282 feat + conv stem)
python New_CNN_Transformer_Train.py        # ablations via env: NEWFEAT=0, CONVSTEM=0, SEED=0 …

# 3. Train + save the M2 meta-label (after any M1 retrain — M2 is paired to the M1 checkpoint)
python btc_meta_label.py                   # 5-seed validation, then saves btc_eth_sol_meta_m2.pkl

# 4. Live signals (auto-fills the data gap, then scores — direction + M2 confidence + size)
python btc_predict_cross_nf.py             # last 24 h (default)
python btc_predict_cross_nf.py --days 7    # last 7 days, or --from/--to a date range
python btc_predict_cross_nf.py --no-update # skip the data refresh

# 5. Backtest the production model · validate M2 drift-robustness
python btc_backtest_cross_nf.py            # BTC test split (default)
python walk_forward_m2.py                  # M2 expanding-window walk-forward
```

Output: `btc_predict_cross_nf_signals.csv` —
`timestamp, close, prob_long, vol_regime, signal, m2_confidence, recommended_size`.

---

## 13. Tech Stack

`PyTorch` · `NumPy` / `pandas` · `scikit-learn` · `matplotlib` · Binance REST API

**Concepts demonstrated:** temporal-CNN + Transformer design (conv tokenizer, RoPE, ALiBi, SE,
asset embedding) · multi-asset pooled training without leakage · triple-barrier labelling ·
market microstructure (OFI, VPIN, Amihud) · cross-asset / PCA-absorption / permutation-entropy
features · focal loss & class imbalance · volatility-regime calibration · **meta-labeling
(direction/sizing split) with a GBM+RandomForest ensemble** · **confidence-scaled position
sizing** · **expanding-window walk-forward validation** · **PSI feature-drift monitoring &
calibration (ECE)** · **seed-controlled A/B experimentation** · multi-seed backtest validation ·
modular train/serve-parity architecture · model interpretability.
