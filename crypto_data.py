"""
crypto_data.py — Unified BTC / ETH / SOL data manager (Binance public klines).

Combines the three separate scripts (cmc_api.py, eth_download.py, sol_download.py)
into one. Two modes:

  update   (default) — download the last DAYS of candles and MERGE into the existing
                       historical CSVs (de-dup + sort + continuity check). Run this
                       routinely before btc_predict.py.
  download           — bulk-paginate the FULL history (2018 → now) and (re)write the
                       CSVs from scratch. Use once to seed a new asset.

No API key required. Timestamp format / columns match the existing CSVs.

Examples:
  py -3.10 crypto_data.py                         # update all assets, all timeframes
  py -3.10 crypto_data.py --assets btc eth        # update only BTC + ETH
  py -3.10 crypto_data.py --mode download         # full history for all assets
  py -3.10 crypto_data.py --mode download --assets sol
  py -3.10 crypto_data.py --days 30               # update with a 30-day overlap window
"""
import os, time, argparse, requests
import pandas as pd
from datetime import datetime, timedelta, timezone

DATA_DIR    = r"D:\Document\LLLLLLLLLLLLL\DATA"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
LIMIT       = 1000
SLEEP       = 0.30
START       = "2018-01-01"          # SOL simply returns from its 2020 listing
UPDATE_DAYS = 10                    # default overlap window for --mode update

CSV_HEADERS = [
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore",
]

ASSETS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}

# (interval, expected delta, filename template)
TIMEFRAMES = [
    ("15m", pd.Timedelta("15min"), "{a}_15m_data_2018_to_2025.csv"),
    ("1h",  pd.Timedelta("1h"),    "{a}_1h_data_2018_to_2025.csv"),
    ("4h",  pd.Timedelta("4h"),    "{a}_4h_data_2018_to_2025.csv"),
    ("1d",  pd.Timedelta("1D"),    "{a}_1d_data_2018_to_2025.csv"),
]


# ─────────────────────────────────────────────
# BINANCE FETCH
# ─────────────────────────────────────────────
def ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f ")


def fetch_range(symbol: str, interval: str, start_ms: int) -> list:
    """Paginate from start_ms to now, dropping the still-forming final candle."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows, req = [], 0
    while start_ms < now_ms:
        resp = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": LIMIT},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        req += 1
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][6] + 1
        if nxt <= start_ms:
            break
        start_ms = nxt
        if len(batch) < LIMIT:
            break
        if req % 25 == 0:
            print(f"        … {req} requests, {len(rows):,} candles "
                  f"(through {ms_to_str(batch[-1][0]).strip()})")
        time.sleep(SLEEP)
    if rows and rows[-1][6] > now_ms:
        rows = rows[:-1]
    return rows


def rows_to_df(rows: list) -> pd.DataFrame:
    data = [
        [ms_to_str(r[0]), r[1], r[2], r[3], r[4], r[5],
         ms_to_str(r[6]), r[7], r[8], r[9], r[10], r[11]]
        for r in rows
    ]
    return pd.DataFrame(data, columns=CSV_HEADERS)


# ─────────────────────────────────────────────
# CSV LOAD / CONTINUITY
# ─────────────────────────────────────────────
def load_historical(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["Open time"] = df["Open time"].str.strip().str.replace(" UTC", "", regex=False)
    df["_ts"] = pd.to_datetime(df["Open time"], utc=True, errors="coerce")
    return df[df["_ts"].notna()].copy()


def check_continuity(df: pd.DataFrame, expected: pd.Timedelta, new_cutoff=None) -> None:
    diffs = df["_ts"].diff()
    gaps = df[diffs > expected * 1.5]
    if gaps.empty:
        print("        continuity : OK"); return
    if new_cutoff is not None:
        new_gaps = gaps[gaps["_ts"] >= new_cutoff]
        old_gaps = gaps[gaps["_ts"] <  new_cutoff]
        if not new_gaps.empty:
            print(f"        continuity : WARNING — {len(new_gaps)} gap(s) in NEW window")
        else:
            print("        continuity : OK — new window gap-free")
        if not old_gaps.empty:
            print(f"        old gaps   : {len(old_gaps)} pre-existing (Binance downtime)")
    else:
        print(f"        gaps       : {len(gaps)} pre-existing (Binance downtime, 2018-20)")


def reformat_save(merged: pd.DataFrame, path: str) -> None:
    merged = merged.sort_values("_ts").drop_duplicates("_ts").reset_index(drop=True)
    merged["Open time"] = merged["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S.%f ")
    merged["Close time"] = pd.to_datetime(
        merged["Close time"].str.strip().str.replace(" UTC", "", regex=False),
        utc=True, errors="coerce",
    ).dt.strftime("%Y-%m-%d %H:%M:%S.%f ")
    merged[CSV_HEADERS].to_csv(path, index=False)


# ─────────────────────────────────────────────
# PER ASSET / TIMEFRAME
# ─────────────────────────────────────────────
def process(asset: str, symbol: str, interval: str, expected: pd.Timedelta,
            path: str, mode: str, days: int) -> None:
    exists = os.path.exists(path)
    do_full = (mode == "download") or (not exists)

    if do_full:
        tag = "download" if mode == "download" else "seed (no file yet)"
        print(f"  [{asset} {interval:>3s}] full {tag} …")
        rows = fetch_range(symbol, interval, int(pd.Timestamp(START, tz="UTC").timestamp() * 1000))
        new_df = rows_to_df(rows)
        new_df["_ts"] = pd.to_datetime(new_df["Open time"].str.strip(), utc=True, errors="coerce")
        reformat_save(new_df, path)
        saved = load_historical(path)
        print(f"        saved : {len(saved):,} rows  "
              f"({saved['_ts'].iloc[0]:%Y-%m-%d %H:%M} -> {saved['_ts'].iloc[-1]:%Y-%m-%d %H:%M})")
        check_continuity(saved, expected)
    else:
        print(f"  [{asset} {interval:>3s}] update (last {days}d) …")
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        rows = fetch_range(symbol, interval, start_ms)
        new_df = rows_to_df(rows)
        new_df["_ts"] = pd.to_datetime(new_df["Open time"].str.strip(), utc=True, errors="coerce")
        new_cutoff = new_df["_ts"].iloc[0] if len(new_df) else None
        hist = load_historical(path)
        before = len(hist)
        merged = pd.concat([hist, new_df], ignore_index=True)
        reformat_save(merged, path)
        saved = load_historical(path)
        print(f"        merged : {before:,} + {len(saved)-before} new = {len(saved):,} rows  "
              f"(-> {saved['_ts'].iloc[-1]:%Y-%m-%d %H:%M})")
        check_continuity(saved, expected, new_cutoff)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Unified BTC/ETH/SOL Binance data manager")
    ap.add_argument("--mode", choices=["update", "download"], default="update",
                    help="update = merge recent into existing CSVs; download = full history rewrite")
    ap.add_argument("--assets", nargs="+", default=list(ASSETS.keys()),
                    choices=list(ASSETS.keys()), help="which assets (default: all)")
    ap.add_argument("--days", type=int, default=UPDATE_DAYS, help="overlap window for --mode update")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Mode: {args.mode}   assets: {', '.join(args.assets)}\n")
    for asset in args.assets:
        symbol = ASSETS[asset]
        print(f"=== {symbol} ===")
        for interval, expected, tmpl in TIMEFRAMES:
            path = os.path.join(DATA_DIR, tmpl.format(a=asset))
            process(asset, symbol, interval, expected, path, args.mode, args.days)
            time.sleep(0.25)
        print()
    print("All done. Run btc_predict.py for updated signals.")


if __name__ == "__main__":
    main()
