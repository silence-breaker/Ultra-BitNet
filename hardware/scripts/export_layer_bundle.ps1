param(
    [int]$FirstLayer = 0,
    [int]$Layers = 1,
    [int]$ActivationSeed = 23
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..\..")
$exportTool = Join-Path $rootDir "tools/export_layer_bundle.py"
$outDir = Join-Path $rootDir "build/model"

if ($env:BITNET_MODEL_FILE) {
    $modelFile = $env:BITNET_MODEL_FILE
} else {
    $contestDir = "G:/" +
        [char]0x96C6 + [char]0x521B + [char]0x533A + [char]0x57DF +
        [char]0x603B + [char]0x51B3 + [char]0x8D5B
    $deployDir = "AXU3EGB" +
        [char]0x6A21 + [char]0x578B + [char]0x90E8 + [char]0x7F72
    $modelFile = Join-Path $contestDir (Join-Path $deployDir "bitnet-b1.58-2B-4T-deploy/model.safetensors")
}

if (-not (Test-Path $exportTool)) {
    throw "Missing layer export tool: $exportTool"
}
if (-not (Test-Path $modelFile)) {
    throw "Missing model file: $modelFile"
}

$python = $env:PYTHON
if (-not $python) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

& $python `
    $exportTool `
    --model $modelFile `
    --out-dir $outDir `
    --first-layer $FirstLayer `
    --layers $Layers `
    --activation-seed $ActivationSeed

exit $LASTEXITCODE
