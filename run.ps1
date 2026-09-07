# Clinical Agent - create venv, install deps, run the FastAPI app.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host 'Creating virtual environment (.venv)...'
    python -m venv .venv
}

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -r requirements.txt

$model = if ($env:CLAUDE_MODEL) { $env:CLAUDE_MODEL } else { 'claude-sonnet-5' }
$engine = if ($env:ANTHROPIC_API_KEY) { "synthesis ($model)" } else { 'evidence-only (no ANTHROPIC_API_KEY)' }
Write-Host ""
Write-Host "Clinical Agent  ->  http://127.0.0.1:8000"
Write-Host "Engine mode     ->  $engine"
Write-Host ""

& $py -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
