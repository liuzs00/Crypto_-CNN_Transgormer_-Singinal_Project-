"""
funding_data.py — Download crypto-native derivatives positioning data (Binance USDⓈ-M perps).

Two orthogonal, non-price signals the spot OHLCV pipeline cannot reconstruct:
  - FUNDING RATE  (fapi/v1/fundingRate) — the 8h perpetual funding charge. Persistent positive
    funding = crowded longs (carry cost + squeeze risk); negative = crowded shorts. Pure
    positioning / sentiment, settled every 00:00 / 08:00 / 16:00 UTC.
  - PERP BASIS    (perp 4h close vs spot 4h close) — leverage demand / contango-backwardation.

Binance perpetual futures launched AFTER the 2018 spot history starts, so the early rows have
no funding/basis and are neutral-filled (flag `fund_avail=0`) by crypto_features.add_funding:
  BTCUSDT perp: 2019-09-25   ·   ETHUSDT perp: 2019-11-27   ·   SOLUSDT perp: 2020-09-14
The 2024–25 test split has full coverage. Gold (PAXG) has no perp → skipped (macro-only asset).

Saves per asset:  DATA/{a}_funding.csv  (timestamp, funding_rate)
                  DATA/{a}_perp_4h.csv (timestamp, Close)   <- perp close for the basis leg

Run:  py -3.10 funding_data.py                 # btc eth sol: full funding + perp-4h history
      py -3.10 funding_data.py --assets btc     # single asset
      py -3.10 funding_data.py --days 30         # update tail only (merge into existing CSVs)
"""
import os, time, argparse, requests
import pandas as pd
from datetime import datetime, timezone, timedelta

from paths import DATA_DIR
FAPI      = "https://fapi.binance.com"
SYMBOLS   = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}   # PAXG has no perp
START     = "2019-09-01"          # just before the earliest perp (BTC) listing
SLEEP     = 0.25


def _get(path, params):
    r = requests.get(FAPI + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_funding(symbol, start_ms):
    """Paginate /fapi/v1/fundingRate (limit 1000, 8h spacing) from start_ms to now."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000); rows = []
    while start_ms < now_ms:
        batch = _get("/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": start_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1]["fundingTime"] + 1
        if nxt <= start_ms or len(batch) < 1000:
            start_ms = nxt; break
        start_ms = nxt; time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return (df[["timestamp", "funding_rate"]].dropna()
            .drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True))


def fetch_perp_klines(symbol, interval, start_ms):
    """Paginate /fapi/v1/klines (limit 1500) — keep open-time + close for the basis leg."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000); rows = []
    while start_ms < now_ms:
        batch = _get("/fapi/v1/klines",
                     {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= start_ms or len(batch) < 1500:
            break
        start_ms = nxt; time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "Close"])
    df = pd.DataFrame({"timestamp": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
                       "Close": pd.to_numeric([r[4] for r in rows], errors="coerce")})
    return df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _merge_save(new_df, path, key="timestamp"):
    if new_df.empty:
        print(f"        no data → {os.path.basename(path)} unchanged"); return
    if os.path.exists(path):
        old = pd.read_csv(path); old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
        new_df = pd.concat([old.dropna(subset=[key]), new_df], ignore_index=True)
    new_df = new_df.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    new_df.to_csv(path, index=False)
    print(f"        saved : {len(new_df):>6,} rows  ({new_df[key].iloc[0]:%Y-%m-%d} -> "
          f"{new_df[key].iloc[-1]:%Y-%m-%d})  -> {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=list(SYMBOLS.keys()), choices=list(SYMBOLS.keys()))
    ap.add_argument("--days", type=int, default=None, help="update tail: fetch only last N days and merge")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    if args.days:
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)
    else:
        start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    for a in args.assets:
        sym = SYMBOLS[a]; print(f"=== {sym} ({a}) ===")
        print("  funding rate …")
        _merge_save(fetch_funding(sym, start_ms), os.path.join(DATA_DIR, f"{a}_funding.csv"))
        print("  perp 4h klines …")
        _merge_save(fetch_perp_klines(sym, "4h", start_ms), os.path.join(DATA_DIR, f"{a}_perp_4h.csv"))
        print()
    print("Done. Feature build (crypto_features.add_funding) forward-fills onto the 4h grid + avail flag.")


if __name__ == "__main__":
    main()
