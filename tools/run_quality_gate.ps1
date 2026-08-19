$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([string]$Label) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Label failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

$root = (& git rev-parse --show-toplevel 2>$null)
Assert-NativeSuccess "Locate repository"
if (-not $root) {
    Write-Error "Run this script from inside the Open-LineCaller repository."
    exit 2
}

Set-Location $root

Write-Host "Open LineCaller strict quality gate"
Write-Host "Repository: $root"
Write-Host ""

python -m pytest -q
Assert-NativeSuccess "Full pytest suite"

git diff --check
Assert-NativeSuccess "git diff --check"

Write-Host ""
Write-Host "QUALITY GATE PASSED"
exit 0