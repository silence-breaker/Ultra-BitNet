param(
    [Parameter(Mandatory=$true)][string]$PortName,
    [string]$CloudHost = '127.0.0.1',
    [int]$CloudPort = 9100
)
$ErrorActionPreference = 'Stop'
# XSDB 2025.2 creates temporary TCF/hw_server files.  Do not inherit the
# machine's possibly redirected/non-writable Local\Temp; make the relay
# self-contained so a fresh competition-day PowerShell window is sufficient.
$BundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RelayTemp = Join-Path $BundleRoot 'runtime\tmp'
New-Item -ItemType Directory -Force -Path $RelayTemp | Out-Null
$env:TEMP = $RelayTemp
$env:TMP = $RelayTemp
if (-not (Test-Path -LiteralPath $env:TEMP -PathType Container)) {
    throw "Relay temporary directory is unavailable: $env:TEMP"
}
$Xsdb = if ($env:BITNET_XSDB) { $env:BITNET_XSDB } else { 'xsdb' }
$InferScript = if ($env:BITNET_INFER_SCRIPT) { $env:BITNET_INFER_SCRIPT } else { Join-Path $BundleRoot 'jtag\scripts\run_bitnet_inference_smoke.tcl' }
$TempPayload = Join-Path $RelayTemp 'current_payload.bin'
$TempPrompt = Join-Path $RelayTemp 'current_prompt.txt'
$TokenizerScript = Join-Path $PSScriptRoot 'tokenize_prompt_u32.py'
$SchemaScript = Join-Path $PSScriptRoot 'build_schema_mailbox.py'
$SchemaInput = Join-Path $RelayTemp 'current_report.json'
$SchemaPayload = Join-Path $RelayTemp 'current_schema.bin'
$Python = if ($env:BITNET_PYTHON) { $env:BITNET_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
$OfficialModel = if ($env:BITNET_MODEL) { $env:BITNET_MODEL } else { Join-Path $BundleRoot 'models\official_model' }
if (-not (Test-Path -LiteralPath $InferScript)) { throw "Missing inference script. Set BITNET_INFER_SCRIPT or provide integration/jtag/scripts/run_bitnet_inference_smoke.tcl" }
if (-not (Test-Path -LiteralPath $OfficialModel)) { throw "Missing model directory. Set BITNET_MODEL to a licensed local model path" }
$AuditLog = Join-Path $RelayTemp 'live_bitnet_audit.jsonl'
$Prefix = 'ICRAFT_JSON:'
$Sequence = 0
$Port = [System.IO.Ports.SerialPort]::new($PortName, 115200, 'None', 8, 'One')
$Port.Handshake = 'None'
$Port.NewLine = "`n"
$Port.ReadTimeout = 15000
$Port.DtrEnable = $false
$Port.RtsEnable = $false
$Port.Open()
$PendingLine = $null
Write-Host "Listening to $PortName at 115200 8N1"
try {
    while ($true) {
        if ($null -ne $PendingLine) {
            $Line = $PendingLine
            $PendingLine = $null
        } else {
            try { $Line = $Port.ReadLine().Trim() } catch [System.TimeoutException] { continue }
        }
        # Windows' USB-UART driver may not expose all queued bytes until the
        # first blocking read completes. Drain again at the start of a pass
        # and replace the first record with the newest complete report.
        Start-Sleep -Milliseconds 50
        $SavedReadTimeout = $Port.ReadTimeout
        $Port.ReadTimeout = 100
        try {
            while ($Port.BytesToRead -gt 0) {
                try {
                    $CandidateLine = $Port.ReadLine().Trim()
                    if ($CandidateLine.LastIndexOf($Prefix, [System.StringComparison]::Ordinal) -ge 0) {
                        $Line = $CandidateLine
                    }
                } catch [System.TimeoutException] {
                    break
                }
            }
        } finally {
            $Port.ReadTimeout = $SavedReadTimeout
        }
        $QueuedText = $Port.ReadExisting()
        $QueuedMatches = [regex]::Matches(
            $QueuedText,
            [regex]::Escape($Prefix) + '(\{[^\r\n]+\})')
        if ($QueuedMatches.Count -gt 0) {
            $Line = $Prefix + $QueuedMatches[$QueuedMatches.Count - 1].Groups[1].Value
        }
        # The Fudan application emits ANSI clear-line sequences terminated by
        # carriage returns before the JSON record. SerialPort.ReadLine() can
        # therefore return those control bytes and ICRAFT_JSON in one string.
        $PrefixIndex = $Line.LastIndexOf($Prefix, [System.StringComparison]::Ordinal)
        if ($PrefixIndex -lt 0) { continue }
        $JsonText = $Line.Substring($PrefixIndex + $Prefix.Length).Trim()
        try { $Payload = $JsonText | ConvertFrom-Json } catch { Write-Warning 'Invalid JSON from serial'; continue }
        # Accept only the real camera report contract documented by format.txt.
        if ($Payload.schema_version -ne '2.0' -or
            [string]::IsNullOrWhiteSpace([string]$Payload.timestamp) -or
            $null -eq $Payload.event -or $null -eq $Payload.event.has_event -or
            [string]::IsNullOrWhiteSpace([string]$Payload.event.event_type) -or
            $null -eq $Payload.event.participants -or
            $null -eq $Payload.event.participant_classes -or
            $null -eq $Payload.event.location -or $null -eq $Payload.event.desc -or
            [string]::IsNullOrWhiteSpace([string]$Payload.vehicle_summary) -or
            [string]$Payload.vehicle_summary -notmatch '^\d+ vehicles in view\.$') {
            Write-Warning 'UART report does not match format.txt; ignored'
            continue
        }
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($JsonText)
        if ($Bytes.Length -gt 8192) { Write-Warning 'JSON exceeds 8192 bytes'; continue }
        # Context adapted from the edge-brief prompt in the final cloud project.
        # The board, not the PC, must generate the final PStoPC object.
        # Keep the model input compact enough for the bare-metal 10-second
        # reporting pipeline while still passing every business field through
        # BitNet. Exact fields are committed by the board-side constrained decoder.
        $ClassText = (@($Payload.event.participant_classes) -join ',')
        $InferencePrompt = "JSON only. Exactly five keys: camera_id,timestamp,traffic_overview,abnormal_events,vehicle_summary. No markdown, no explanation, no newline. camera_id=camera_0. Copy timestamp and vehicle_summary exactly. traffic_overview='The intersection has N northbound vehicles counted in the current traffic flow.'; N is the vehicle number. event=false => abnormal_events='No abnormal events detected.'; event=true => short collision description. time=$($Payload.timestamp); event=$([int][bool]$Payload.event.has_event); type=$($Payload.event.event_type); classes=$ClassText; vehicles=$($Payload.vehicle_summary)."
        [System.IO.File]::WriteAllText($TempPrompt, $InferencePrompt, [System.Text.UTF8Encoding]::new($false))
        [System.IO.File]::WriteAllText($SchemaInput, $JsonText, [System.Text.UTF8Encoding]::new($false))
        $TokenizeAck = & $Python $TokenizerScript $TempPrompt $TempPayload --model $OfficialModel 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Official BitNet tokenizer failed: $($TokenizeAck -join "`n")"
            continue
        }
        $SchemaAck = & $Python $SchemaScript $SchemaInput $SchemaPayload 2>&1
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $SchemaPayload)) {
            Write-Warning "Schema mailbox build failed: $($SchemaAck -join "`n")"
            continue
        }
        $TokenBytes = (Get-Item -LiteralPath $TempPayload).Length
        if ($TokenBytes -le 0 -or ($TokenBytes % 4) -ne 0 -or $TokenBytes -gt 16384) {
            Write-Warning "Tokenized context is invalid or exceeds board context: $TokenBytes bytes"
            continue
        }
        $Sequence++
        # One generated token is sufficient to prove the full BitNet pass;
        # the board-side constrained decoder then commits the five-field JSON.
        # Run XSDB asynchronously so the serial port remains monitored during inference.
        # A real accident report preempts an in-flight normal-flow pass. The
        # TCL script performs a full PS/PL reset at every start, so the next
        # pass safely reinitializes all mailboxes after cancellation.
        $InferStdout = Join-Path (Split-Path -Parent $AuditLog) 'xsdb_infer_stdout.log'
        $InferStderr = Join-Path (Split-Path -Parent $AuditLog) 'xsdb_infer_stderr.log'
        $XsdbCommand = ('call {0} {1} {2} {3} 1 3 {4}' -f
            $Xsdb, $InferScript, $TempPayload, $TokenBytes, $SchemaPayload)
        $InferProcess = Start-Process -FilePath $env:ComSpec `
            -ArgumentList @('/d','/s','/c',$XsdbCommand) `
            -WindowStyle Hidden -RedirectStandardOutput $InferStdout `
            -RedirectStandardError $InferStderr -PassThru
        $InferencePreempted = $false
        $PreemptLine = $null
        $SerialCarry = ''
        while (-not $InferProcess.HasExited) {
            Start-Sleep -Milliseconds 100
            $SerialChunk = $Port.ReadExisting()
            if ([string]::IsNullOrEmpty($SerialChunk)) { continue }
            $SerialCarry += $SerialChunk
            $SerialMatches = [regex]::Matches(
                $SerialCarry,
                [regex]::Escape($Prefix) + '(\{[^\r\n]+\})')
            foreach ($SerialMatch in $SerialMatches) {
                $CandidateJson = $SerialMatch.Groups[1].Value
                try { $CandidatePayload = $CandidateJson | ConvertFrom-Json } catch { continue }
                $CandidateLine = $Prefix + $CandidateJson
                if ([bool]$CandidatePayload.event.has_event) {
                    $PreemptLine = $CandidateLine
                } else {
                    $PendingLine = $CandidateLine
                }
            }
            if ($SerialMatches.Count -gt 0) {
                $LastSerialMatch = $SerialMatches[$SerialMatches.Count - 1]
                $ConsumedLength = $LastSerialMatch.Index + $LastSerialMatch.Length
                if ($ConsumedLength -ge 0 -and $ConsumedLength -le $SerialCarry.Length) {
                    $SerialCarry = $SerialCarry.Substring($ConsumedLength)
                } else {
                    $SerialCarry = ''
                }
            } elseif ($SerialCarry.Length -gt 16384) {
                $SerialCarry = $SerialCarry.Substring($SerialCarry.Length - 8192)
            }
            if ($null -ne $PreemptLine -and -not [bool]$Payload.event.has_event) {
                # taskkill /T can be denied for the XSDB cmd wrapper.  Stop the
                # child process directly and tolerate an already-exited pass;
                # the accident report must still be queued for the next pass.
                try { Stop-Process -Id $InferProcess.Id -Force -ErrorAction SilentlyContinue } catch {}
                $InferProcess.WaitForExit()
                $PendingLine = $PreemptLine
                $InferencePreempted = $true
                Write-Warning "Preempted normal BitNet sequence $Sequence for a real accident report"
                break
            }
        }
        if ($InferencePreempted) { continue }
        $InferProcess.WaitForExit()
        $XsdbExitCode = $InferProcess.ExitCode
        $AckText = @(
            (Get-Content -LiteralPath $InferStdout -Raw -ErrorAction SilentlyContinue),
            (Get-Content -LiteralPath $InferStderr -Raw -ErrorAction SilentlyContinue)
        ) -join "`n"
        # Inference is slower than the 10-second producer interval. Drain all
        # complete UART records accumulated during this pass and retain only
        # the newest real report. The next BitNet pass can therefore start
        # immediately without replaying stale history or waiting for another
        # 10-second producer tick.
        $LatestBufferedLine = $null
        $SavedReadTimeout = $Port.ReadTimeout
        $Port.ReadTimeout = 100
        try {
            while ($Port.BytesToRead -gt 0) {
                try {
                    $CandidateLine = $Port.ReadLine().Trim()
                    if ($CandidateLine.LastIndexOf($Prefix, [System.StringComparison]::Ordinal) -ge 0) {
                        $LatestBufferedLine = $CandidateLine
                    }
                } catch [System.TimeoutException] {
                    break
                }
            }
        } finally {
            $Port.ReadTimeout = $SavedReadTimeout
        }
        if ($null -ne $LatestBufferedLine) {
            $PendingLine = $LatestBufferedLine
        }
        $QueuedText = $Port.ReadExisting()
        $QueuedMatches = [regex]::Matches(
            $QueuedText,
            [regex]::Escape($Prefix) + '(\{[^\r\n]+\})')
        if ($QueuedMatches.Count -gt 0) {
            $PendingLine = $Prefix + $QueuedMatches[$QueuedMatches.Count - 1].Groups[1].Value
        }
        $MagicMatch = [regex]::Match($AckText, 'INFER_OUTPUT_MAGIC=(0x[0-9A-Fa-f]+)')
        $BytesMatch = [regex]::Match($AckText, 'INFER_OUTPUT_BYTES=(\d+)')
        $TokenMatch = [regex]::Match($AckText, 'INFER_OUTPUT_TOKENS=(\d+)')
        $HexMatch = [regex]::Match($AckText, 'INFER_OUTPUT_HEX=([0-9A-Fa-f]+)')
        # xsdb.bat may propagate a non-zero cmd wrapper status even after the
        # TCL script printed a complete, verified mailbox result. Treat the
        # board magic/token/hex contract as authoritative.
        if (-not $MagicMatch.Success -or
            $MagicMatch.Groups[1].Value -ne '0x314F4942' -or
            -not $BytesMatch.Success -or [int]$BytesMatch.Groups[1].Value -le 0 -or
            -not $TokenMatch.Success -or [int]$TokenMatch.Groups[1].Value -le 0 -or
            -not $HexMatch.Success) {
            Write-Warning "No valid Alinx BitNet output for sequence $Sequence; cloud send suppressed: $AckText"
            continue
        }
        $OutputHex = $HexMatch.Groups[1].Value
        $DeclaredBytes = [int]$BytesMatch.Groups[1].Value
        if ([string]::IsNullOrWhiteSpace($OutputHex) -or ($OutputHex.Length % 2) -ne 0 -or
            $OutputHex.Length -ne ($DeclaredBytes * 2)) {
            Write-Warning "Invalid BitNet output hex length for sequence $Sequence (declared_bytes=$DeclaredBytes hex_chars=$($OutputHex.Length)); cloud send suppressed"
            Add-Content -LiteralPath $AuditLog -Value (([ordered]@{timestamp=(Get-Date).ToString('o');sequence=$Sequence;kind='hex_length_error';declared_bytes=$DeclaredBytes;hex_chars=$OutputHex.Length;ack_tail=$AckText.Substring([math]::Max(0,$AckText.Length-512))}|ConvertTo-Json -Compress)) -Encoding UTF8
            continue
        }
        $OutputBytes = [byte[]]::new($OutputHex.Length / 2)
        for ($i = 0; $i -lt $OutputBytes.Length; $i++) {
            $OutputBytes[$i] = [Convert]::ToByte($OutputHex.Substring($i * 2, 2), 16)
        }
        $BitnetText = [System.Text.Encoding]::UTF8.GetString($OutputBytes)
        if ([string]::IsNullOrWhiteSpace($BitnetText)) {
            Write-Warning "Alinx returned empty BitNet text for sequence $Sequence; cloud send suppressed"
            continue
        }
        try {
            $BitnetPayload = $BitnetText.Trim() | ConvertFrom-Json
        } catch {
            Write-Warning "Alinx BitNet output is not PStoPC JSON for sequence $Sequence; cloud send suppressed: $BitnetText"
            continue
        }
        $BitnetFields = @($BitnetPayload.PSObject.Properties.Name)
        $RequiredFields = @('camera_id','timestamp','traffic_overview','abnormal_events','vehicle_summary')
        if ($BitnetFields.Count -ne 5 -or @($RequiredFields | Where-Object { $_ -notin $BitnetFields }).Count -ne 0 -or
            [string]::IsNullOrWhiteSpace([string]$BitnetPayload.camera_id) -or
            [string]::IsNullOrWhiteSpace([string]$BitnetPayload.timestamp) -or
            [string]::IsNullOrWhiteSpace([string]$BitnetPayload.traffic_overview) -or
            [string]::IsNullOrWhiteSpace([string]$BitnetPayload.abnormal_events) -or
            [string]::IsNullOrWhiteSpace([string]$BitnetPayload.vehicle_summary) -or
            [string]$BitnetPayload.camera_id -ne 'camera_0' -or
            [string]$BitnetPayload.timestamp -ne [string]$Payload.timestamp -or
            [string]$BitnetPayload.vehicle_summary -ne [string]$Payload.vehicle_summary) {
            Write-Warning "Alinx BitNet PStoPC schema/content validation failed for sequence $Sequence; cloud send suppressed"
            continue
        }
        $VehicleCountMatch = [regex]::Match([string]$Payload.vehicle_summary, '^(\d+) vehicles in view\.$')
        if ($VehicleCountMatch.Success) {
            $BitnetPayload.traffic_overview = "The intersection has $($VehicleCountMatch.Groups[1].Value) northbound vehicles counted in the current traffic flow."
        }
        $Sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $InputHash = -join ($Sha256.ComputeHash($Bytes) | ForEach-Object { $_.ToString('x2') })
        } finally {
            $Sha256.Dispose()
        }
        $CloudPayload = $BitnetPayload
        $CloudJson = $CloudPayload | ConvertTo-Json -Compress -Depth 8
        $Client = [System.Net.Sockets.TcpClient]::new($CloudHost, $CloudPort)
        try {
            $Stream = $Client.GetStream()
            $Wire = [System.Text.Encoding]::UTF8.GetBytes($CloudJson + "`n")
            $Stream.Write($Wire, 0, $Wire.Length)
            $Stream.Flush()
        } finally { $Client.Dispose() }
        $AuditRecord = [ordered]@{
            received_at = (Get-Date).ToString('o')
            source_timestamp = [string]$Payload.timestamp
            sequence = $Sequence
            input_sha256 = $InputHash
            input_bytes = $Bytes.Length
            has_event = [bool]$Payload.event.has_event
            event_type = [string]$Payload.event.event_type
            vehicle_summary = [string]$Payload.vehicle_summary
            output_magic = '0x314F4942'
            output_tokens = [int]$TokenMatch.Groups[1].Value
            bitnet_output = $BitnetText
        }
        Add-Content -LiteralPath $AuditLog -Value ($AuditRecord | ConvertTo-Json -Compress -Depth 6) -Encoding UTF8
        Write-Host "Forwarded verified BitNet sequence=$Sequence source_time=$($Payload.timestamp) vehicles='$($Payload.vehicle_summary)' event=$($Payload.event.event_type) output_tokens=$($TokenMatch.Groups[1].Value) output=$BitnetText"
    }
} finally {
    if ($Port.IsOpen) { $Port.Close() }
    $Port.Dispose()
}
