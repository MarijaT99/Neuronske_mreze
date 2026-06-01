#requires -Version 5
<#
.SYNOPSIS
    Robust local-run wrapper for plant-disease-hierarchical.

.DESCRIPTION
    Pins the working directory to the repo root (this script's folder) before
    invoking Python, so relative paths work even when PowerShell's CWD has
    drifted to the parent directory — a recurring problem when the repo lives
    under OneDrive with a non-ASCII path. Also sets PYTHONDONTWRITEBYTECODE so
    Python does not litter __pycache__ into the synced tree.

    Everything after the script/module is forwarded to Python untouched.

.EXAMPLE
    .\run.ps1 scripts\sanity_check.py

.EXAMPLE
    .\run.ps1 scripts\prepare_splits.py --data-root "data\raw\plantvillage dataset\color" --out-dir data\splits --seed 42

.EXAMPLE
    .\run.ps1 -m pytest

.NOTES
    Set $env:PDH_PYTHON to override the interpreter. Otherwise a local
    .venv\Scripts\python.exe is preferred, falling back to "python" on PATH.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Forward
)

# Always run from the repo root (the directory containing this script).
Set-Location -LiteralPath $PSScriptRoot

# Keep the OneDrive-synced tree free of compiled bytecode.
$env:PYTHONDONTWRITEBYTECODE = '1'

$python =
    if ($env:PDH_PYTHON) { $env:PDH_PYTHON }
    elseif (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' }
    else { 'python' }

& $python @Forward
exit $LASTEXITCODE
