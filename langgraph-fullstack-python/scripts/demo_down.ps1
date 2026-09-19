<#
.SYNOPSIS
    Stop the PsycheGraph demo stack (container path A).

.DESCRIPTION
    `demo_down.ps1` only stops and removes the containers. The Hugging Face
    weights volume (~4.5 GB, one-time download) and the host-side RAG index
    under data/vectorstore are kept, so the next start is fast.
    Use -WipeVolumes to delete the weights volume as well: the next start will
    re-download BGE-M3 inside the container (about 14 minutes).

.EXAMPLE
    scripts\demo_down.ps1
    scripts\demo_down.ps1 -WipeVolumes
#>
[CmdletBinding()]
param(
    [switch]$WipeVolumes
)

$ErrorActionPreference = 'Continue'

$composeFile = 'deploy/docker-compose.dev.yml'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $composeFile)) { throw "Cannot find $composeFile - is this the repository root?" }

if ($WipeVolumes) {
    Write-Host '==> Stopping the stack and deleting the HF weights volume' -ForegroundColor Cyan
    cmd /c "docker compose -f $composeFile --env-file .env down -v"
} else {
    Write-Host '==> Stopping the stack (volumes and index are kept)' -ForegroundColor Cyan
    cmd /c "docker compose -f $composeFile --env-file .env down"
}

if ($LASTEXITCODE -ne 0) { throw 'docker compose down failed - see the output above.' }

Write-Host ''
Write-Host 'Stack stopped.' -ForegroundColor Green
if ($WipeVolumes) {
    Write-Host 'The HF weights volume was removed: the next start re-downloads BGE-M3.' -ForegroundColor Yellow
} else {
    Write-Host 'HF weights volume and data/vectorstore were kept - the next start takes seconds.' -ForegroundColor DarkGray
}
