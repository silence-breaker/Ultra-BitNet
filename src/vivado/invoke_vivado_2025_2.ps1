param(
    [string]$VivadoRoot = 'D:\Vivado\install\2025.2\Vivado',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$VivadoArguments
)

$ErrorActionPreference = 'Stop'
$vivadoExe = Join-Path $VivadoRoot 'bin\unwrapped\win64.o\vivado.exe'
if (-not (Test-Path -LiteralPath $vivadoExe)) {
    throw "Vivado executable not found: $vivadoExe"
}

$installVersionRoot = Split-Path -Parent $VivadoRoot
$installRoot = Split-Path -Parent $installVersionRoot
$savedPath = $env:PATH
$savedTclLibrary = $env:TCL_LIBRARY
$savedTkLibrary = $env:TK_LIBRARY
$savedRdiDataDir = $env:RDI_DATADIR
$savedRtLibPath = $env:RT_LIBPATH
$savedRtTclPath = $env:RT_TCL_PATH
$savedSynthCommon = $env:SYNTH_COMMON
$savedHrtTclPath = $env:HRT_TCL_PATH

try {
    $env:RDI_APPROOT = $VivadoRoot.Replace('\', '/')
    $env:HDI_APPROOT = $env:RDI_APPROOT
    $env:RDI_BASEROOT = $installVersionRoot.Replace('\', '/')
    $env:RDI_BINROOT = (($VivadoRoot + '\bin').Replace('\', '/'))
    $env:RDI_INSTALLROOT = $installRoot.Replace('\', '/')
    $env:RDI_INSTALLVERSION = Split-Path -Leaf $installVersionRoot
    $env:RDI_INSTALLVER = 'Vivado'
    $env:RDI_PLATFORM = 'win64'
    $env:RDI_OPT_EXT = '.o'
    $env:RDI_LIBDIR = (($VivadoRoot + '\lib\win64.o').Replace('\', '/'))
    $env:XILINX_VIVADO = $env:RDI_APPROOT
    $sharedData = Join-Path $installVersionRoot 'data\shareddata'
    $env:RDI_SHARED_DATA = $sharedData.Replace('\', '/')
    $env:RDI_DATADIR = $env:RDI_SHARED_DATA + ';' +
        (($VivadoRoot + '\data').Replace('\', '/'))
    $env:TCL_LIBRARY = (($installVersionRoot + '\tps\tcl\tcl8.6').Replace('\', '/'))
    $env:TK_LIBRARY = (($installVersionRoot + '\tps\tk\tk8.6').Replace('\', '/'))

    # Vivado's normal loader sets these before vivado.exe loads xv_synth.dll.
    # Directly launching the unwrapped executable without RT_LIBPATH makes the
    # synthesis database paths collapse to /svlog/sdbs and /vhdl/.../vdbs.
    $rtLibPath = Join-Path $VivadoRoot 'scripts\rt\data'
    $rtTclPath = Join-Path $VivadoRoot 'scripts\rt\base_tcl\tcl'
    $hrtTclPath = Join-Path $VivadoRoot 'scripts\rt\fpga_tcl'
    foreach ($requiredPath in @($rtLibPath, $rtTclPath, $hrtTclPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Vivado real-time synthesis data not found: $requiredPath"
        }
    }
    $env:RT_LIBPATH = $rtLibPath.Replace('\', '/')
    $env:RT_TCL_PATH = $rtTclPath.Replace('\', '/')
    $env:SYNTH_COMMON = $env:RT_LIBPATH
    $env:HRT_TCL_PATH = $hrtTclPath.Replace('\', '/')

    $runtimePaths = @(
        (Join-Path $VivadoRoot 'bin'),
        (Join-Path $VivadoRoot 'bin\unwrapped\win64.o'),
        (Join-Path $VivadoRoot 'lib\win64.o'),
        (Join-Path $VivadoRoot 'tps\win64'),
        (Join-Path $VivadoRoot 'tps\win64\python-3.13.0'),
        (Join-Path $VivadoRoot 'tps\win64\python-3.13.0\DLLs'),
        (Join-Path $VivadoRoot 'tps\win64\jre21.0.5_11\bin'),
        (Join-Path $VivadoRoot 'tps\win64\jre21.0.5_11\bin\server')
    )
    $env:PATH = ($runtimePaths -join ';') + ';' + $savedPath

    & $vivadoExe @VivadoArguments
    exit $LASTEXITCODE
}
finally {
    $env:PATH = $savedPath
    $env:TCL_LIBRARY = $savedTclLibrary
    $env:TK_LIBRARY = $savedTkLibrary
    $env:RDI_DATADIR = $savedRdiDataDir
    $env:RT_LIBPATH = $savedRtLibPath
    $env:RT_TCL_PATH = $savedRtTclPath
    $env:SYNTH_COMMON = $savedSynthCommon
    $env:HRT_TCL_PATH = $savedHrtTclPath
}
