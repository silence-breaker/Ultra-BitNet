param([string]$PortName='COM7')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RelayDir=$PSScriptRoot
$Tmp=Join-Path $Root 'runtime\tmp'
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
$env:TEMP=$Tmp; $env:TMP=$Tmp
$CloudDir=Join-Path $Root 'cloud'
$Xsdb=(Get-ChildItem -LiteralPath 'D:\icraft2\2025.2\Vitis\bin\xsdb.bat' | Select-Object -ExpandProperty FullName)
$Probe=Join-Path $Tmp 'wait_psu.tcl'
@('connect -url tcp:127.0.0.1:3121','jtag targets 1','puts [targets]','disconnect') | Set-Content -LiteralPath $Probe -Encoding ascii
function Test-Port([int]$p){ try { return (Test-NetConnection 127.0.0.1 -Port $p -InformationLevel Quiet -WarningAction SilentlyContinue) } catch { return $false } }
if(-not (Test-Port 9100)){ Start-Process powershell.exe -ArgumentList @('-NoProfile','-Command',"Set-Location -LiteralPath '$CloudDir'; & python .\server.py") -WindowStyle Hidden }
for($i=0;$i -lt 30 -and -not (Test-Port 9100);$i++){Start-Sleep 1}
if(-not (Test-Port 9100)){throw 'Cloud TCP 9100 did not start'}
if(-not ([System.IO.Ports.SerialPort]::GetPortNames() -contains $PortName)){throw "$PortName is not present"}
# Do not start until the PS debug domain is visible; PL-only/DAP states cause
# the old relay to launch XSDB and lose the report immediately.
$psu=$false
for($i=0;$i -lt 30 -and -not $psu;$i++){
  $txt=& $Xsdb $Probe 2>&1 | Out-String
  $psu=$txt -match 'PSU'
  if(-not $psu){Start-Sleep 2}
}
if(-not $psu){throw 'JTAG PSU/A53 targets are not visible'}
$relay=Join-Path $RelayDir 'serial_jtag_cloud_relay.ps1'
Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$relay,'-PortName',$PortName,'-CloudHost','127.0.0.1','-CloudPort','9100') -WindowStyle Hidden
Write-Host "FULL_CHAIN_READY cloud=9100 serial=$PortName jtag=PSU-visible temp=$Tmp"
