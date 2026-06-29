@echo off
REM Wrapper that the scheduled task runs every 4h: auto-updates data, scores the latest
REM signal, and appends the output to predict_cron.log. The "LATEST SIGNAL -> ..." line in
REM that log (and the last row of btc_predict_cross_nf_signals.csv) is your call.
cd /d D:\Document\LLLLLLLLLLLLL
echo. >> predict_cron.log
echo ===== run %DATE% %TIME% (local) ===== >> predict_cron.log
py -3.10 btc_predict_cross_nf.py --days 1 >> predict_cron.log 2>&1
py -3.10 notify_telegram.py >> predict_cron.log 2>&1
