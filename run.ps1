#requires -Version 5
<#
.SYNOPSIS
    Robustan wrapper za lokalno pokretanje plant-disease-hierarchical projekta.

.DESCRIPTION
    Fiksira radni direktorijum na koren repozitorijuma (folder ove skripte) pre
    pokretanja Python-a, tako da relativne putanje rade čak i kada je CWD u
    PowerShell-u odlutao do roditeljskog direktorijuma — što je čest problem kada
    repozitorijum živi pod OneDrive-om sa putanjom koja sadrži ne-ASCII znakove.
    Takođe postavlja PYTHONDONTWRITEBYTECODE da Python ne zatrpava __pycache__-om
    sinhronizovano stablo.

    Sve nakon skripte/modula prosleđuje se Python-u nepromenjeno.

.EXAMPLE
    .\run.ps1 scripts\sanity_check.py

.EXAMPLE
    .\run.ps1 scripts\prepare_splits.py --data-root "data\raw\plantvillage dataset\color" --out-dir data\splits --seed 42

.EXAMPLE
    .\run.ps1 -m pytest

.NOTES
    Postavi $env:PDH_PYTHON da prepišeš interpreter. U suprotnom, prednost ima
    lokalni .venv\Scripts\python.exe, uz fallback na "python" sa PATH-a.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Forward
)

# Uvek pokreni iz korena repozitorijuma (direktorijum koji sadrži ovu skriptu).
Set-Location -LiteralPath $PSScriptRoot

# Drži stablo sinhronizovano sa OneDrive-om bez kompajliranog bytecode-a.
$env:PYTHONDONTWRITEBYTECODE = '1'

$python =
    if ($env:PDH_PYTHON) { $env:PDH_PYTHON }
    elseif (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' }
    else { 'python' }

& $python @Forward
exit $LASTEXITCODE
