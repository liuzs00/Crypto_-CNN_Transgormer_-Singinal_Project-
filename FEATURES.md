# Feature Dictionary — Newest Model (`emb_cross_nf`)

Model: `btc_eth_sol_pooled_model_emb_cross_nf.pth` · Trainer: `btc_eth_sol_cross_train.py`
**274 features** per candle, all **scale-invariant** (ratios / returns / z-scores) and
strictly **causal** (no look-ahead).

## Structure (how 274 is composed)

| Block | Count | Source |
|---|---:|---|
| Per-timeframe indicators × 4 timeframes (15m, 1h, 4h, 1d) | ~65 × 4 = 260 | `add_indicators()` |
| Cross-timeframe divergence | 4 | base merge |
| Cross-asset (BTC vs ETH/SOL) + systemic | 10 | `add_cross_asset()` + `add_absorption()` |
| **Total** | **274** | |

Each per-timeframe feature below exists **four times**, prefixed `15m_ 1h_ 4h_ 1d_`
(e.g. `4h_rsi14`, `1d_rsi14`). The regime/threshold logic keys off `4h_atr14`.

---

## 1. Microstructure / Order Flow
Derived from raw exchange fields (taker-buy volume, trade count, quote volume).

| Feature | Formula | Meaning |
|---|---|---|
| `buy_ratio` | taker_buy / volume ∈ [0,1] | aggressor buy pressure |
| `buy_ratio_ma` | 10-bar MA of buy_ratio | smoothed buy pressure |
| `buy_ratio_dev` | buy_ratio − 20-bar MA | deviation from recent norm |
| `delta_vol` | 2·buy_ratio − 1 ∈ [−1,1] | signed flow imbalance |
| `delta_vol_ma` | 10-bar MA of delta_vol | smoothed signed flow |
| `ofi5 / ofi10 / ofi20` | Σ(buy−sell) / Σvol over 5/10/20 | **Order-Flow Imbalance** (net signed flow) |
| `ofi_mom` | ofi5 − ofi20 | acceleration of order flow |
| `vpin5 / vpin20` | Σ\|buy−sell\| / Σvol over 5/20 | **VPIN** — flow toxicity / informed trading |
| `ofi_div` | 10-bar MA of [sign(ret) − sign(ofi)] | price-up-but-sellers-dominate → bearish divergence |
| `amihud` | \|ret\| / quote_vol, normalized | **Amihud illiquidity** — price impact per dollar |
| `trade_int` | n_trades / 20-bar MA | trade intensity vs normal |
| `trade_size` | (vol/n_trades) / 20-bar MA | avg trade size (retail vs institutional footprint) |

## 2. Volatility & Jump Structure

| Feature | Formula | Meaning |
|---|---|---|
| `atr14 / atr7` | ATR(14 or 7) / Close | normalized true range |
| `atr_r` | ATR7 / ATR14 | short-vs-long vol ratio (vol acceleration) |
| `rvol10 / rvol20` | std(returns) over 10/20 | realized volatility |
| `bbw` | (BB_upper − BB_lower) / BB_mid | Bollinger bandwidth (vol regime) |
| `bbp` | (Close − BB_lower)/(BB_upper − BB_lower) | %B — position in the band |
| `vol_sync` | (H−L)/volume, normalized | volume-synchronized volatility (impact per unit volume) |
| `jump_ratio` | RV / RBV, where RV=Σr², RBV=(π/2)·Σ\|rₜ\|\|rₜ₋₁\| | **realized-bipower jump ratio** — >1 flags a discontinuous jump |
| `upper_sweep` | (upper wick / ATR) × volume-ratio | bull liquidity-sweep / stop-hunt signature |
| `lower_sweep` | (lower wick / ATR) × volume-ratio | bear liquidity-sweep signature |
| `range_eff` | \|ret\| / ((H−L)/Close) | range efficiency (1 = directional, ~0 = inside-out sweep) |

## 3. Trend (EMA cascade)

| Feature | Formula | Meaning |
|---|---|---|
| `pr9 / pr21 / pr50 / pr200` | Close / EMA(n) − 1 | price vs each EMA |
| `ema9_21` | EMA9/EMA21 − 1 | fast trend alignment |
| `ema21_50` | EMA21/EMA50 − 1 | mid trend alignment |
| `ema50_200` | EMA50/EMA200 − 1 | long trend alignment (golden/death-cross proxy) |

## 4. Momentum / Oscillators

| Feature | Formula | Meaning |
|---|---|---|
| `rsi7 / rsi14` | RSI(n) / 100 ∈ [0,1] | momentum / overbought-oversold |
| `macd` | (EMA12 − EMA26) / ATR | ATR-normalized MACD line |
| `macds` | EMA9(macd) | MACD signal line |
| `macdh` | macd − macds | MACD histogram (momentum shift) |
| `stk` | stochastic %K = (C−LL14)/(HH14−LL14) | position in 14-bar range |
| `std` | 3-bar MA of %K | smoothed stochastic |
| `wpr` | Williams %R = (HH14−C)/(HH14−LL14) | overbought (0) / oversold (1) |
| `cci` | Commodity Channel Index, scaled | deviation from typical price |

## 5. Ichimoku Cloud

| Feature | Formula | Meaning |
|---|---|---|
| `ichi_tk` | (Tenkan − Kijun) / Close | TK-cross momentum signal |
| `ichi_pos` | (Close − cloud_mid) / Close | price vs cloud center |
| `ichi_cld` | (SenkouA − SenkouB) / Close | cloud color & thickness |

## 6. Volume

| Feature | Formula | Meaning |
|---|---|---|
| `vr` | volume / 20-bar MA | relative volume |
| `vr5` | 5-bar MA / 20-bar MA | short-term volume surge |
| `obv` | On-Balance-Volume, 30-bar z-score | cumulative volume-flow trend |

## 7. Returns & Distribution Shape

| Feature | Formula | Meaning |
|---|---|---|
| `ret1 / ret2 / ret3 / ret6 / ret12 / ret24` | pct_change(n) | multi-horizon returns |
| `skew20 / skew10` | rolling skewness of returns | asymmetry (crash vs melt-up risk) |
| `kurt20` | rolling kurtosis of returns | fat-tailedness |
| `autocorr` | lag-1 autocorrelation of returns | momentum (+) vs mean-reversion (−) |

## 8. Candle Structure

| Feature | Formula | Meaning |
|---|---|---|
| `body` | (Close − Open) / ATR | candle body size & direction |
| `hl` | (High − Low) / Close | candle range |
| `upper` | upper wick / ATR | rejection from above |
| `lower` | lower wick / ATR | rejection from below |

## 9. Anchored VWAP Distance — **new feature**

Stationary z-score of price vs the volume-weighted institutional cost basis, anchored
to structural events. Restores the "where is price" information that pure ratios strip out.

| Feature | Formula | Meaning |
|---|---|---|
| `vwap_dist_w` | (Close − weekly-anchored VWAP) / ATR | distance from this week's cost basis (mean-reversion gravity) |
| `vwap_dist_m` | (Close − monthly-anchored VWAP) / ATR | distance from this month's cost basis |

> aVWAP = cumulative Σ(typical_price·volume) / Σvolume, reset at each week/month open.

---

## 10. Cross-Timeframe Divergence
Captures conflict between fast and slow horizons (computed once, on the 4h base).

| Feature | Formula | Meaning |
|---|---|---|
| `div_rsi_4h1d` | 4h_rsi14 − 1d_rsi14 | 4h vs daily momentum divergence |
| `div_rsi_4h1h` | 4h_rsi14 − 1h_rsi14 | 4h vs hourly momentum divergence |
| `div_macdh_4h1d` | 4h_macdh − 1d_macdh | 4h vs daily MACD-histogram divergence |
| `div_vol_4h1d` | 4h_vr / 1d_vr | 4h vs daily relative-volume ratio |

---

## 11. Cross-Asset Features (BTC vs ETH/SOL)
Symmetric "this asset vs the **rest** of the universe" — leave-one-out market reference,
so the columns mean the same thing for every asset (keeps pooled training valid).

| Feature | Formula | Meaning |
|---|---|---|
| `x_relret1` | ret1(this) − mean(ret1 of others) | relative strength, short |
| `x_relret6` | ret6(this) − mean(ret6 of others) | relative strength, mid |
| `x_relret24` | ret24(this) − mean(ret24 of others) | relative strength, long |
| `x_relvol` | ATR(this) / mean(ATR of others) | relative volatility — who's driving risk |
| `x_relrsi` | RSI(this) − mean(RSI of others) | relative momentum |
| `x_beta` | 60-bar cov(r, market_r) / var(market_r) | β to the rest of the universe |
| `x_corr` | 60-bar corr(r, market_r) | co-movement regime |
| `x_rank6` | cross-sectional percentile rank of 6-bar return | relative ranking across assets |

## 12. Systemic Fragility — **new feature**
Market-wide scalar (same value for all assets) from rolling PCA of the BTC/ETH/SOL
return covariance.

| Feature | Formula | Meaning |
|---|---|---|
| `x_absorb` | λ₁ / Σλ of the 60-bar return covariance (PCA) | **Absorption Ratio** — variance share of the 1st principal component. High = assets in lockstep (macro-driven, fragile; idiosyncratic signals matter less). Low = micro-driven (assets on their own fundamentals). |
| `x_absorb_chg` | 6-bar change in `x_absorb` | absorption momentum — **spikes precede market-wide liquidations/rallies** |

---

## Design Notes

- **Scale invariance:** every feature is a ratio, return, or z-score — stationary across
  BTC's 10×+ price range and comparable across BTC/ETH/SOL (enabling pooled training).
- **Causality / no leak:** rolling windows look only backward; cross-asset and absorption
  features use *contemporaneous* (≤ t) values of the other assets, never future data.
- **Normalization:** `RobustScaler` (5–95 quantile) fit on the **training partition only**,
  outputs clipped to ±6.
- **Inference dependency:** because of the cross-asset + absorption features, BTC inference
  with this model requires **live ETH and SOL data** (BTC-only inference is not possible).
- **The two newest features** (`vwap_dist_*`, `x_absorb*`) gave the project's largest single
  accuracy jump — but only on the multi-asset model with enough data to learn them; they
  overfit the smaller BTC-only baseline.
