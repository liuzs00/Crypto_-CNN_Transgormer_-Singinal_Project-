# BTC Multi-Timeframe Temporal Transformer — Long/Short Signal Engine

A research-grade deep-learning pipeline that forecasts directional trading signals for
Bitcoin (BTC-USDT) over a 4-hour horizon. The system fuses **four timeframes**
(15m / 1h / 4h / 1d), engineers **256 scale-invariant market-microstructure and
technical features**, and trains a custom **Temporal Transformer** (RoPE + ALiBi
positional encoding, Squeeze-and-Excitation gating, patch embedding) with a
**triple-barrier labelling scheme** from quantitative finance.

> The goal is not "predict the price" — it is to frame trading as a **probabilistic,
> risk-aware classification problem** and to engineer the full stack end-to-end:
> data ingestion → feature engineering → labelling → model → calibration → inference.

---

## 1. Problem Formulation

Naïve price-direction prediction (`up` vs `down` next candle) ignores **risk/reward**
and **path dependency**. Instead, each candle is labelled with the **Triple-Barrier
Method** (López de Prado, *Advances in Financial Machine Learning*):

| Label | Meaning | Condition (looking forward ≤ 4 candles) |
|:---:|---|---|
| **1** (Long) | Take-profit hit first | High reaches **+1.5%** before Low reaches −3.0% |
| **0** (Short) | Stop-loss hit first | Low reaches **−3.0%** before High reaches +1.5% |
| **NaN** | Ambiguous / timeout | both barriers in one candle, or neither within 4 candles → **excluded** |

This encodes a concrete, asymmetric **risk-reward target** (1.5% TP vs 3.0% SL) directly
into the learning objective. The asymmetry also produces a realistic class imbalance
(~73% Long / 27% Short in a multi-year bull market), which the training pipeline handles
explicitly with class weighting and focal loss.

---

## 2. Architecture

```
            ┌──────────────────────────────────────────────────────────┐
            │  4 timeframes (15m / 1h / 4h / 1d)  →  256 features        │
            │  aligned to the 4h grid via backward merge-asof           │
            └──────────────────────────────────────────────────────────┘
                                      │  sequence of 64 candles (≈11 days)
                                      ▼
        Squeeze-and-Excitation gate   →  learns which feature channels matter
                                      ▼
        Patch Embedding (4 steps→1 token)  →  16 tokens, denoises micro-jitter
                                      ▼
        ┌───────────── Temporal Transformer × 3 layers ─────────────┐
        │   Pre-LayerNorm  ·  8-head attention                      │
        │   RoPE  (relative position via Q/K rotation)              │
        │   ALiBi (linear distance recency bias, parameter-free)    │
        │   GELU feed-forward (d_ff = 512)                          │
        └───────────────────────────────────────────────────────────┘
                                      ▼
                 Mean-pool over tokens  →  2-layer MLP head  →  P(Long), P(Short)
```

- **~1.9M parameters**, `d_model = 256`, sequence length 64, patch size 4.
- **RoPE + ALiBi** are combined deliberately: RoPE encodes *relative position* inside the
  attention dot-product, while ALiBi adds an explicit *recency* penalty — both are
  parameter-free and well-suited to non-stationary financial series where recent context
  dominates.
- **Squeeze-and-Excitation** performs global feature-channel recalibration, letting the
  network down-weight noisy indicators per-sample.

---

## 3. Feature Engineering (256 features)

All features are **scale-invariant** (ratios, returns, z-scores, ATR-normalised) so the
model generalises across BTC's 10×+ price range from 2018–2026.

**Market microstructure / order flow** (from raw exchange fields)
- Taker buy-pressure ratio & deviation; signed **Order-Flow Imbalance (OFI)** at 5/10/20 windows
- **VPIN** proxy (flow toxicity), OFI momentum, OFI–price divergence
- **Amihud illiquidity**, volume-synchronised volatility, trade intensity & average trade size

**Volatility & jump structure**
- ATR (7/14) ratios, Bollinger bandwidth/%B, realised volatility (10/20)
- **Realised Bipower Variation** jump ratio (RV/RBV) — separates diffusion from jumps
- Liquidity-sweep detection (wick × volume spike), range efficiency

**Trend / momentum / oscillators**
- EMA ratio cascade (9/21/50/200), MACD (ATR-normalised), RSI (7/14)
- Stochastic, Williams %R, CCI, Ichimoku cloud (TK cross, price-vs-cloud, cloud depth)
- OBV z-score, multi-horizon returns, rolling skew/kurtosis, return autocorrelation

**Cross-timeframe divergence**
- 4h-vs-1d and 4h-vs-1h RSI/MACD divergence, 4h-vs-1d volume ratio — captures
  regime conflict between fast and slow horizons.

---

## 4. Training Methodology

| Component | Choice | Rationale |
|---|---|---|
| Loss | **Focal loss + label smoothing (0.1)** | focus on hard examples; avoid overconfident plateau |
| Class weighting | inverse-frequency (configurable Short factor) | counter the 73/27 Long/Short imbalance |
| Augmentation | Gaussian jitter + random feature masking | regularise against noisy, non-stationary inputs |
| Schedule | LR warmup → cosine decay | stable early training, smooth convergence |
| Split | **chronological** 70 / 15 / 15 | no look-ahead leakage; scaler fit on train only |
| Scaling | `RobustScaler` (5–95 quantile) + clip ±6 | robust to fat-tailed crypto outliers |
| Early stopping | patience on validation loss | prevent overfitting |

**Leakage discipline:** sequences are built from the *full* row matrix so that the
temporal context at **training time is identical to inference time**, and the feature
scaler is fit strictly on the training partition.

---

## 5. Volatility-Regime-Adaptive Thresholding

A single probability cutoff is suboptimal across volatility regimes. The engine classifies
each candle into **low / mid / high** ATR regimes (percentile boundaries learned on the
training set) and applies regime-specific LONG/SHORT confidence gates:

| Regime | Long thr | Short thr | Intuition |
|---|:---:|:---:|---|
| Low vol | 0.55 | 0.45 | calmer tape → looser gate |
| Mid vol | 0.60 | 0.40 | balanced |
| High vol | 0.72 | 0.28 | demand high conviction in chaos |

A dedicated calibration script (`btc_threshold_tuning.py`) performs a **3-stage grid search**
(regime boundaries → per-regime thresholds → joint fine-tune) optimising a
**precision-weighted objective** and an **expected-PnL** proxy
(`prec·TP − (1−prec)·SL`), subject to minimum-coverage constraints.

---

## 6. Representative Results (held-out test set)

| Metric | Value |
|---|---|
| Directional accuracy (argmax) | **~0.70** |
| **Long** precision | **~0.82** |
| Short precision | ~0.45–0.50 |
| Confidence-filtered accuracy | **~0.72** |

The model is intentionally **stronger on Long than Short** — the asymmetric barriers make
a −3% down-move twice as hard to trigger as a +1.5% up-move, and shorts are rarer in a bull
market. This is surfaced honestly rather than hidden, and is exactly the kind of
class-conditional behaviour a risk desk needs to understand before sizing positions.

> *Disclaimer: research project for skill demonstration. Not financial advice and not a
> live trading system.*

---

## 7. Engineering & Interpretability

- **Attention-map analysis**: column-sum "received attention" per patch, per-layer entropy,
  recency ratios, and Long-vs-Short attention profiles — used to verify the model attends to
  recent context and to compare correct vs incorrect predictions.
- **Reproducible inference**: training and `btc_predict.py` produce **bit-identical
  probabilities** for the same timestamp (verified to < 1e-7), with regime boundaries
  serialised into the checkpoint so calibration never drifts between train and serve.
- **Automated data layer** (`cmc_api.py`): pulls fresh Binance klines, drops the still-forming
  candle, **merges into the historical CSVs with de-duplication and continuity checks** across
  all four timeframes.

---

## 8. Repository Structure

```
├── cmc_api.py                # Binance updater: fetch → de-dup → continuity check → merge
├── btc_lstm_train.py         # Feature engineering + Temporal Transformer training
├── btc_predict.py            # Inference CLI (default last 24h, or any date range)
├── btc_threshold_tuning.py   # 3-stage regime/threshold calibration
├── btc_lstm_tuning.py        # Sequential hyperparameter search (staged, resumable)
├── btc_lstm_tuning_r2.py     # Second-round refinement search
├── DATA/                     # Multi-timeframe OHLCV (2018 → present)
└── btc_lstm_model.pth        # Trained checkpoint (weights + scaler + feat cols + regime bounds)
```

---

## 9. Usage

```bash
# 1. Refresh data (merges latest Binance candles into the historical CSVs)
python cmc_api.py

# 2. Train the model
python btc_lstm_train.py

# 3. Generate signals
python btc_predict.py                              # last 24 hours (default)
python btc_predict.py --days 7                      # last 7 days
python btc_predict.py --from 2026-06-01             # from a date to latest
python btc_predict.py --from 2026-06-01 --to 2026-06-10
python btc_predict.py --from "2026-06-01 08:00" --to "2026-06-03 20:00"

# 4. (Optional) Calibrate regime thresholds
python btc_threshold_tuning.py
```

Output: `btc_recent_signals.csv` — `timestamp, prob_long, prob_short, signal, vol_regime, thresholds`.

---

## 10. Tech Stack

`PyTorch` · `NumPy` / `pandas` · `scikit-learn` · `matplotlib` · Binance REST API

**Concepts demonstrated:** Transformer architecture design (RoPE, ALiBi, SE, patch
embedding) · focal loss & class imbalance · time-series cross-validation without leakage ·
market microstructure (OFI, VPIN, Amihud) · triple-barrier labelling · volatility-regime
modelling · probability calibration · model interpretability.
