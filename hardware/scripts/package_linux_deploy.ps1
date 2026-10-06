param(
    [string]$OutputDirectory = "",
    [ValidateSet("stable", "plddr-dual", "pspl-dual")]
    [string]$HardwareVariant = "stable"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $Root "build/bitnet_linux_deploy"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

if ($HardwareVariant -eq "pspl-dual") {
    $Hardware = @{
        Bitstream = "$Root/build/bitnet_accel_pspl_dual.bit"
        Xsa = "$Root/build/axu3egb_pspl_dual.xsa"
        Timing = "$Root/build/pspl_dual_timing_summary_routed.rpt"
        Utilization = "$Root/build/pspl_dual_utilization_placed.rpt"
    }
} elseif ($HardwareVariant -eq "plddr-dual") {
    $Hardware = @{
        Bitstream = "$Root/build/bitnet_accel_plddr_dual.bit"
        Xsa = "$Root/build/axu3egb_plddr_dual.xsa"
        Timing = "$Root/build/plddr_dual_timing_summary_routed.rpt"
        Utilization = "$Root/build/plddr_dual_utilization_placed.rpt"
    }
} else {
    $Hardware = @{
        Bitstream = "$Root/build/vivado/axu3egb_bitnet_accel.runs/impl_1/design_1_wrapper.bit"
        Xsa = "$Root/build/axu3egb_bitnet_accel.xsa"
        Timing = "$Root/build/vivado/axu3egb_bitnet_accel.runs/impl_1/design_1_wrapper_timing_summary_routed.rpt"
        Utilization = "$Root/build/vivado/axu3egb_bitnet_accel.runs/impl_1/design_1_wrapper_utilization_placed.rpt"
    }
}
foreach ($Required in @($Hardware.Bitstream, $Hardware.Xsa, $Hardware.Timing, $Hardware.Utilization)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Hardware artifact missing for variant '$HardwareVariant': $Required"
    }
}

if (Test-Path -LiteralPath $OutputDirectory) {
    Remove-Item -LiteralPath $OutputDirectory -Recurse -Force
}
foreach ($Directory in @($OutputDirectory, "$OutputDirectory/firmware", "$OutputDirectory/model", "$OutputDirectory/baremetal", "$OutputDirectory/bin")) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

Copy-Item -LiteralPath "$Root/hardware/linux/compat" -Destination "$OutputDirectory/compat" -Recurse
foreach ($Name in @("bitnet_linux.c", "bitnet_client.c", "bitnet_client.h", "bitnet_plddr_dual.h", "prompt_source.c", "prompt_source.h", "Makefile", "install.sh", "bitnet-split.sh", "bitnet-reserved-memory.dtsi", "bitstream.bif")) {
    Copy-Item -LiteralPath "$Root/hardware/linux/$Name" -Destination $OutputDirectory
}
$LinuxBinary = "$Root/hardware/linux/build/bitnet-linux"
$RuntimeSources = @(
    Get-ChildItem -LiteralPath "$Root/hardware/linux" -File |
        Where-Object { $_.Extension -in @(".c", ".h") -or $_.Name -eq "Makefile" }
    Get-ChildItem -LiteralPath "$Root/hardware/linux/compat" -File
    Get-ChildItem -LiteralPath "$Root/hardware/baremetal" -File |
        Where-Object { $_.Extension -in @(".c", ".h") }
)
$LatestRuntimeSource = $RuntimeSources |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if ((Test-Path -LiteralPath $LinuxBinary) -and
    ((Get-Item -LiteralPath $LinuxBinary).LastWriteTimeUtc -ge $LatestRuntimeSource.LastWriteTimeUtc)) {
    Copy-Item -LiteralPath $LinuxBinary -Destination "$OutputDirectory/bin/bitnet-linux"
} else {
    Write-Warning "AArch64 executable missing or stale; install.sh will rebuild it on the target Linux system."
}
Copy-Item -LiteralPath "$Root/READMD.md" -Destination $OutputDirectory
foreach ($Source in Get-ChildItem -LiteralPath "$Root/hardware/baremetal" -File | Where-Object { $_.Extension -in @(".c", ".h") }) {
    Copy-Item -LiteralPath $Source.FullName -Destination "$OutputDirectory/baremetal"
}

& tar.exe -cf "$OutputDirectory/model/BITNET.tar" -C "$Root/build/emmc" BITNET
if ($LASTEXITCODE -ne 0) { throw "model tar failed: $LASTEXITCODE" }
Copy-Item -LiteralPath $Hardware.Bitstream -Destination "$OutputDirectory/firmware/design_1_wrapper.bit"
Copy-Item -LiteralPath $Hardware.Xsa -Destination "$OutputDirectory/firmware"
Copy-Item -LiteralPath "$Root/build/vitis_ws/bitnet_accel_smoke/_ide/psinit/psu_init.tcl" -Destination "$OutputDirectory/firmware"
Copy-Item -LiteralPath $Hardware.Timing -Destination "$OutputDirectory/firmware"
Copy-Item -LiteralPath $Hardware.Utilization -Destination "$OutputDirectory/firmware"

$Bootgen = "G:/Vivado_2020.1/Vivado/2020.1/bin/bootgen.bat"
if (-not (Test-Path -LiteralPath $Bootgen)) {
    throw "bootgen executable missing: $Bootgen"
}
$PreviousProcessorArchitecture = $env:PROCESSOR_ARCHITECTURE
Push-Location $OutputDirectory
try {
    # Xilinx 2020.1 Windows launchers silently return without doing any work
    # when PROCESSOR_ARCHITECTURE is absent from a non-interactive process.
    $env:PROCESSOR_ARCHITECTURE = "AMD64"
    & $Bootgen -image bitstream.bif -arch zynqmp -process_bitstream bin -w
    if ($LASTEXITCODE -ne 0) { throw "bootgen failed: $LASTEXITCODE" }
    $GeneratedBitBin = "firmware/design_1_wrapper.bit.bin"
    $PackagedBitBin = "firmware/bitnet_accel.bit.bin"
    if (-not (Test-Path -LiteralPath $GeneratedBitBin)) {
        throw "bootgen returned success but did not create $GeneratedBitBin"
    }
    Move-Item -LiteralPath $GeneratedBitBin -Destination $PackagedBitBin -Force
    if (-not (Test-Path -LiteralPath $PackagedBitBin)) {
        throw "FPGA Manager firmware missing after bootgen: $PackagedBitBin"
    }
} finally {
    if ($null -eq $PreviousProcessorArchitecture) {
        Remove-Item Env:PROCESSOR_ARCHITECTURE -ErrorAction SilentlyContinue
    } else {
        $env:PROCESSOR_ARCHITECTURE = $PreviousProcessorArchitecture
    }
    Pop-Location
}

$Files = Get-ChildItem -LiteralPath $OutputDirectory -Recurse -File
$Bytes = ($Files | Measure-Object Length -Sum).Sum
Write-Host "Linux deploy package: $OutputDirectory"
Write-Host "Hardware variant: $HardwareVariant"
Write-Host "Files: $($Files.Count), bytes: $Bytes"
