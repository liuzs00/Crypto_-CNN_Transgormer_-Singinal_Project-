"""
cmc_api.py — BTC candle updater (Binance public API -> historical CSVs)

Every run:
  1. Downloads the last DAYS of BTCUSDT candles for all 4 timeframes
  2. Drops the still-forming (incomplete) last candle
  3. Merges directly into the canonical historical CSV files
  4. Deduplicates and sorts
  5. Reports rows added, current range, and any continuity gaps

No API key required. Run before btc_predict.py to get fresh signals.
"""

import os, time, requests
import pandas as pd
from datetime import datetime, timedelta, timezone

DATA_DIR    = r"D:\Document\LLLLLLLLLLLLL\DATA"
SYMBOL      = "BTCUSDT"
DAYS        = 10          # overlap window — keeps enough history to close any gap
BINANCE_URL = "https://api.binance.com/api/v3/klines"

CSV_HEADERS = [
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore",
]

# (interval, expected_timedelta, historical_csv_filename)
TIMEFRAMES = [
    ("15m", pd.Timedelta("15min"), "btc_15m_data_2018_to_2025.csv"),
    ("1h",  pd.Timedelta("1h"),    "btc_1h_data_2018_to_2025.csv"),
    ("4h",  pd.Timedelta("4h"),    "btc_4h_data_2018_to_2025.csv"),
    ("1d",  pd.Timedelta("1D"),    "btc_1d_data_2018_to_2025.csv"),
]


# ─────────────────────────────────────────────
# BINANCE FETCH
# ─────────────────────────────────────────────
def ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f ")


def fetch_klines(interval: str) -> list:
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
    resp = requests.get(
        BINANCE_URL,
        params={"symbol": SYMBOL, "interval": interval, "startTime": start_ms, "limit": 1000},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    # Drop the last row if its close time is still in the future (candle not yet closed)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
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
# CSV LOAD
# ─────────────────────────────────────────────
def load_historical(path: str) -> pd.DataFrame:
    """Load historical CSV as strings, parse timestamps, drop corrupt rows."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["Open time"] = df["Open time"].str.strip().str.replace(" UTC", "", regex=False)
    df["_ts"] = pd.to_datetime(df["Open time"], utc=True, errors="coerce")
    return df[df["_ts"].notna()].copy()


# ─────────────────────────────────────────────
# CONTINUITY CHECK
# ─────────────────────────────────────────────
def check_continuity(df: pd.DataFrame, expected: pd.Timedelta, new_cutoff: pd.Timestamp) -> None:
    """
    Report gaps in the data. Separates pre-existing historical gaps
    (Binance downtime, 2018-2020) from gaps in the newly downloaded window.
    """
    diffs = df["_ts"].diff()                          # NaN at index 0
    gap_mask = diffs > expected * 1.5
    gaps = df[gap_mask].copy()                        # rows WHERE a gap ends

    if gaps.empty:
        print(f"  continuity : OK")
        return

    old_gaps = gaps[gaps["_ts"] <  new_cutoff]
    new_gaps = gaps[gaps["_ts"] >= new_cutoff]

    if not new_gaps.empty:
        print(f"  continuity : WARNING — {len(new_gaps)} gap(s) in new data window:")
        for idx in new_gaps.index:
            t_before = df.loc[idx - 1, "_ts"].strftime("%Y-%m-%d %H:%M")
            t_after  = df.loc[idx,     "_ts"].strftime("%Y-%m-%d %H:%M")
            size     = diffs.loc[idx]
            print(f"               {t_before}  -->  {t_after}  (missing {size})")
    else:
        print(f"  continuity : OK — new window is gap-free")

    if not old_gaps.empty:
        print(f"  old gaps   : {len(old_gaps)} pre-existing (Binance downtime, not actionable)")


# ─────────────────────────────────────────────
# MERGE + SAVE
# ─────────────────────────────────────────────
def merge_and_save(interval: str, expected: pd.Timedelta, hist_file: str) -> None:
    hist_path = os.path.join(DATA_DIR, hist_file)

    # 1. Fetch
    print(f"[{interval:>3s}]  fetching...", end=" ", flush=True)
    rows = fetch_klines(interval)
    new_df = rows_to_df(rows)
    new_df["_ts"] = pd.to_datetime(new_df["Open time"].str.strip(), utc=True, errors="coerce")
    new_cutoff = new_df["_ts"].iloc[0]
    print(f"{len(rows)} closed candles  "
          f"({new_df['_ts'].iloc[0].strftime('%Y-%m-%d %H:%M')} -> "
          f"{new_df['_ts'].iloc[-1].strftime('%Y-%m-%d %H:%M')})")

    # 2. Load existing
    hist_df    = load_historical(hist_path)
    rows_before = len(hist_df)

    # 3. Merge, dedup, sort
    merged = (pd.concat([hist_df, new_df], ignore_index=True)
                .drop_duplicates(subset=["_ts"])
                .sort_values("_ts")
                .reset_index(drop=True))

    # 4. Re-format timestamps uniformly (YYYY-MM-DD HH:MM:SS.ffffff<space>, no UTC suffix)
    merged["Open time"] = merged["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S.%f ")
    merged["Close time"] = (
        pd.to_datetime(
            merged["Close time"].str.strip().str.replace(" UTC", "", regex=False),
            utc=True, errors="coerce",
        ).dt.strftime("%Y-%m-%d %H:%M:%S.%f ")
    )

    # 5. Save
    merged[CSV_HEADERS].to_csv(hist_path, index=False)

    # 6. Report
    added = len(merged) - rows_before
    print(f"  merged     : {rows_before:,} + {added} new = {len(merged):,} rows")
    print(f"  range      : {merged['_ts'].iloc[0].strftime('%Y-%m-%d %H:%M')}"
          f"  ->  {merged['_ts'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
    check_continuity(merged, expected, new_cutoff)
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Updating {SYMBOL} — last {DAYS} days from Binance -> merging into historical CSVs\n")
    for interval, expected_delta, hist_file in TIMEFRAMES:
        merge_and_save(interval, expected_delta, hist_file)
        time.sleep(0.25)
    print("All done. Run btc_predict.py for updated signals.")
