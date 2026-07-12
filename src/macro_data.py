"""
macro_data.py — Download index-futures macro data (ES, NQ) via yfinance.

Unlike the crypto pairs (Binance, 24/7, 4h back to 2018), index futures come from Yahoo and
have a hard granularity/history limit:
  - 1d (daily): FULL history (2000 -> now)   <- usable for the 2018+ model (forward-filled)
  - 1h        : only the last ~730 days       <- too short to align with 2018-start training
  - <1h       : only the last ~60 days

So the DAILY file is the macro-regime source: forward-filled onto the 4h crypto grid at
feature-build time WITH a market-closed / staleness flag (23/5 futures vs 24/7 crypto). The
recent 1h file is saved too (handy for out-of-sample / recent-period checks) but is NOT
long enough to align with the 2018-start training data.

Run:  py -3.10 macro_data.py                 # ES + NQ: daily (full) + recent 1h
      py -3.10 macro_data.py --daily-only
"""
import os, argparse
import pandas as pd
import yfinance as yf

from paths import DATA_DIR
TICKERS  = {'es': 'ES=F', 'nq': 'NQ=F'}     # S&P 500 + Nasdaq-100 E-mini futures (continuous)
KEEP     = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']


def fetch(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):          # single-ticker download -> ('Open','ES=F') etc.
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={tcol: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df[[c for c in KEEP if c in df.columns]].dropna(subset=['timestamp'])
    return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--daily-only', action='store_true', help='skip the recent-1h files')
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    jobs = [('1d', 'max')] + ([] if args.daily_only else [('1h', '730d')])
    for name, tk in TICKERS.items():
        print(f"=== {tk}  ({name}) ===")
        for interval, period in jobs:
            df = fetch(tk, period, interval)
            if df is None or df.empty:
                print(f"  {interval:>3}: no data"); continue
            path = os.path.join(DATA_DIR, f"{name}_{interval}_data.csv")
            df.to_csv(path, index=False)
            print(f"  {interval:>3}: {len(df):>6,} rows  {df.timestamp.min():%Y-%m-%d} -> "
                  f"{df.timestamp.max():%Y-%m-%d}  -> {os.path.basename(path)}")
    print("\nDone. Daily = macro-regime source (forward-fill onto 4h grid + staleness flag).")


if __name__ == '__main__':
    main()
