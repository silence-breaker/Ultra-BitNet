param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..\..\..")
$exportTool = Join-Path $rootDir "src/tools/export_tokenizer_trie.py"
$outFile = Join-Path $rootDir "build/emmc/BITNET/TOK/TRIE.BIN"

if ($env:BITNET_TOKENIZER_JSON) {
    $tokenizerJson = $env:BITNET_TOKENIZER_JSON
} else {
    throw "Set BITNET_TOKENIZER_JSON to the licensed tokenizer.json path before running this exporter."
}

if (-not (Test-Path $exportTool)) {
    throw "Missing tokenizer export tool: $exportTool"
}
if (-not (Test-Path $tokenizerJson)) {
    throw "Missing tokenizer json: $tokenizerJson"
}

$python = $env:PYTHON
if (-not $python) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

& $python `
    $exportTool `
    --tokenizer-json $tokenizerJson `
    --out $outFile

exit $LASTEXITCODE
