param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
$exportTool = Join-Path $rootDir "tools/export_tokenizer_trie.py"
$outFile = Join-Path $rootDir "build/emmc/BITNET/TOK/TRIE.BIN"

if ($env:BITNET_TOKENIZER_JSON) {
    $tokenizerJson = $env:BITNET_TOKENIZER_JSON
} else {
    $contestDir = "G:/" +
        [char]0x96C6 + [char]0x521B + [char]0x533A + [char]0x57DF +
        [char]0x603B + [char]0x51B3 + [char]0x8D5B
    $deployDir = "AXU3EGB" +
        [char]0x6A21 + [char]0x578B + [char]0x90E8 + [char]0x7F72
    $tokenizerJson = Join-Path $contestDir (Join-Path $deployDir "bitnet-b1.58-2B-4T-deploy/tokenizer.json")
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
