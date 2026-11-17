param([int]$Port = 8501, [string]$PythonExecutable, [switch]$Check)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LOKY_MAX_CPU_COUNT = '1'
if (-not $PythonExecutable) {
    $environmentPython = Join-Path $projectPath '.venv-dashboard\Scripts\python.exe'
    $installedPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
    $runtimeFile = Join-Path $projectPath 'dashboard\runtime.json'
    $preparedPython = if (Test-Path -LiteralPath $runtimeFile) { (Get-Content -LiteralPath $runtimeFile -Raw | ConvertFrom-Json).python_executable } else { $null }
    foreach ($candidate in @($environmentPython, $preparedPython, $installedPython)) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            & $candidate -B -c 'import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(n) is not None for n in ["streamlit","altair","torch","sklearn","pandas"]) else 1)'
            if ($LASTEXITCODE -eq 0) { $PythonExecutable = $candidate; break }
        }
    }
    if (-not $PythonExecutable) { throw 'Use a Python installation with requirements-dashboard.txt installed, as described in dashboard/SETUP.md. No packages are installed automatically.' }
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw 'Python executable was not found.' }
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a local port between 1024 and 65535.' }
Set-Location -LiteralPath $projectPath
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LOKY_MAX_CPU_COUNT = '1'
if ($Check) {
    & $PythonExecutable -B -c 'import streamlit,altair; from dashboard.data import load_dashboard; d=load_dashboard(); print(d["summary"]["model_label"] + " ready: " + str(len(d["captains"])) + " captains; ROC-AUC " + format(d["summary"]["validation"]["roc_auc"], ".6f"))'
    exit $LASTEXITCODE
}
$activeLabel = (Get-Content -LiteralPath (Join-Path $projectPath 'ACTIVE_MODEL.json') -Raw | ConvertFrom-Json).label
Write-Host "Captain Safety Lab — ${activeLabel}: http://127.0.0.1:$Port"
Write-Host 'Keep this terminal running. Press Ctrl+C to stop the dashboard.'
& $PythonExecutable -B -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
exit $LASTEXITCODE
