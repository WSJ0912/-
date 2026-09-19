param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SpecPath = Join-Path $ProjectRoot "services\inference\inference-service.spec"
$DistPath = Join-Path $ProjectRoot "services\inference\dist"
$WorkPath = Join-Path $ProjectRoot "services\inference\build"

$PythonArgs = @()
if (-not $PythonExecutable) {
    $ProjectPythonCandidates = @(
        (Join-Path $ProjectRoot ".venv\python.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    )
    $ProjectPython = $ProjectPythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($ProjectPython) {
        $PythonExecutable = $ProjectPython
    } else {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PyLauncher) {
            $PythonExecutable = $PyLauncher.Source
            $PythonArgs = @("-3.11")
        } else {
            $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
        }
    }
}

& $PythonExecutable @PythonArgs -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 is required for the release service'"
$PythonPrefixBase64 = (& $PythonExecutable @PythonArgs -c "import base64, sys; print(base64.b64encode(sys.prefix.encode('utf-8')).decode('ascii'))").Trim()
try {
    $PythonPrefix = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($PythonPrefixBase64)
    )
} catch {
    throw "Could not decode the selected Python environment path"
}
if (-not $PythonPrefix -or -not (Test-Path -LiteralPath $PythonPrefix)) {
    throw "Could not resolve the selected Python environment"
}

# PyInstaller resolves extension dependencies through PATH on Windows. Keep the
# selected Python environment ahead of a globally activated Conda installation.
$RuntimeSearchPaths = @(
    (Join-Path $PythonPrefix "Library\bin"),
    (Join-Path $PythonPrefix "DLLs"),
    $PythonPrefix
) | Where-Object { Test-Path -LiteralPath $_ }
$OriginalPath = $env:PATH
try {
    $env:PATH = (($RuntimeSearchPaths + @($OriginalPath)) -join [IO.Path]::PathSeparator)
    & $PythonExecutable @PythonArgs -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $DistPath `
        --workpath $WorkPath `
        $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:PATH = $OriginalPath
}

$Executable = Join-Path $DistPath "inference-service\inference-service.exe"
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "PyInstaller did not produce $Executable"
}

$BundledRuntime = Join-Path (Split-Path -Parent $Executable) "_internal"
foreach ($DllName in @("libcrypto-3-x64.dll", "libssl-3-x64.dll")) {
    $SourceDll = @(
        (Join-Path $PythonPrefix "Library\bin\$DllName"),
        (Join-Path $PythonPrefix "DLLs\$DllName"),
        (Join-Path $PythonPrefix $DllName)
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $SourceDll) {
        continue
    }
    $BundledDll = Join-Path $BundledRuntime $DllName
    if (-not (Test-Path -LiteralPath $BundledDll)) {
        throw "PyInstaller omitted required runtime library $DllName"
    }
    $SourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceDll).Hash
    $BundledHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $BundledDll).Hash
    if ($SourceHash -ne $BundledHash) {
        throw "PyInstaller bundled $DllName from outside the selected Python environment"
    }
}
Write-Host "Built $Executable"
