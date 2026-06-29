# register_predict_task.ps1 — register the 4-hourly "CryptoPredict4h" task, fired in UTC.
# RUN ONCE in an *Administrator* PowerShell:
#   powershell -ExecutionPolicy Bypass -File register_predict_task.ps1
#
# Triggers fire at 00:05 / 04:05 / 08:05 / 12:05 / 16:05 / 20:05 **UTC** (note the Z suffix in
# StartBoundary) — i.e. ~5 min after each 4h candle close, the SAME instant worldwide regardless
# of your local timezone. (The Task Scheduler GUI will *display* these converted to your local
# clock — e.g. on UTC+8 you'll see 08:05/12:05/…/04:05 — which is exactly what confirms they are
# UTC-anchored, not local.)

$bat  = "D:\Document\LLLLLLLLLLLLL\run_predict.bat"
$task = "CryptoPredict4h"

# 1. allow wake timers (AC + battery) so a sleeping PC can wake to run the task
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP bd3b718a-0680-4d9d-8ab2-e1d2b4ac806d 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP bd3b718a-0680-4d9d-8ab2-e1d2b4ac806d 1
powercfg /setactive SCHEME_CURRENT

# 2. build six daily UTC triggers  (the trailing 'Z' makes Task Scheduler treat the time as UTC)
$utc = @("00:05","04:05","08:05","12:05","16:05","20:05")
$trigXml = ($utc | ForEach-Object {
@"
    <CalendarTrigger>
      <StartBoundary>2020-01-01T$($_):00Z</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
"@ }) -join "`r`n"

$xml = @"
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Run btc_predict_cross_nf.py every 4h at UTC-aligned times (00:05..20:05 UTC); wakes from sleep.</Description>
  </RegistrationInfo>
  <Triggers>
$trigXml
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <WakeToRun>true</WakeToRun>
    <StartWhenAvailable>true</StartWhenAvailable>
    <Enabled>true</Enabled>
    <ExecutionTimeLimit>PT15M</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>$bat</Command></Exec>
  </Actions>
</Task>
"@

Register-ScheduledTask -TaskName $task -Xml $xml -Force

Write-Host "`nRegistered '$task'. Trigger StartBoundaries (must each end in 'Z' = UTC):"
(Get-ScheduledTask -TaskName $task).Triggers | ForEach-Object { Write-Host "   $($_.StartBoundary)" }
Write-Host "`nNext UTC run shown in your LOCAL time by:  schtasks /query /tn $task /v /fo LIST"
