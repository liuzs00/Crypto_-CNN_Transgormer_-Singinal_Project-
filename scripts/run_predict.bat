@echo off
REM Wrapper that the scheduled task runs every 4h: auto-updates data (Binance + NQ/ES macro),
REM scores the latest signal, appends the run to logs\predict_cron.log, and pushes it to Telegram.
REM The "LATEST SIGNAL -> ..." line in that log (and the last row of
REM outputs\btc_predict_cross_nf_signals.csv) is your call. The append-only signal-state history
REM accumulates in logs\btc_signal_state_log.csv.
REM
REM PORTABLE: resolves the project root relative to THIS script (scripts\ -> ..), so there is no
REM hardcoded machine path — the repo can live anywhere.
pushd "%~dp0.."
if not exist logs mkdir logs
echo. >> logs\predict_cron.log
echo ===== run %DATE% %TIME% (local) ===== >> logs\predict_cron.log
py -3.10 src\btc_predict_cross_nf.py --days 1 >> logs\predict_cron.log 2>&1
py -3.10 src\notify_telegram.py >> logs\predict_cron.log 2>&1
popd
