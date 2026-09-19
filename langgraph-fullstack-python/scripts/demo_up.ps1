<#
.SYNOPSIS
    One-click start for the PsycheGraph demo (container path A).

.DESCRIPTION
    Idempotent: safe to run while the stack is already up. It will
      start Docker Desktop if the engine is down,
      create the stack with docker compose,
      wait until /health answers,
      prewarm the embedding model inside the container (so the first
      question in the demo is fast), then open the browser.
    No business logic lives here - it only orchestrates docker compose.

.EXAMPLE
    scripts\demo_up.ps1
    scripts\demo_up.ps1 -NoBrowser          # start without opening a browser
    scripts\demo_up.ps1 -SkipPrewarm        # skip the ~30 s warm-up step
    scripts\demo_up.ps1 -TimeoutSeconds 300 # slow first start (image pull)

.NOTES
    Messages are intentionally ASCII-only: Windows PowerShell 5.1 reads a
    BOM-less UTF-8 script as ANSI, which would garble Chinese output.
#>
[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipPrewarm,
    [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = 'Continue'

$composeFile = 'deploy/docker-compose.dev.yml'
$baseUrl = 'http://127.0.0.1:8123'

function Write-Step([string]$text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok([string]$text) { Write-Host "    ok  $text" -ForegroundColor Green }
function Write-Note([string]$text) { Write-Host "    !!  $text" -ForegroundColor Yellow }

# --- 1) always work from the repository root --------------------------------
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Step "Repository: $repoRoot"

if (-not (Test-Path $composeFile)) { throw "Cannot find $composeFile - is this the repository root?" }
if (-not (Test-Path '.env')) {
    Write-Note '.env is missing: copy .env.example to .env and set DEEPSEEK_API_KEY.'
    throw '.env not found'
}

# The .env guard: an empty key (or an .env that is a byte-for-byte copy of
# .env.example) makes every run fail with "本次分析未完成，请重试" only after the
# user has typed a question. Fail fast here instead, and say exactly what to do.
function Get-EnvFileValue([string]$name) {
    $line = Get-Content '.env' | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -First 1
    if (-not $line) { return '' }
    return ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
}

$apiKey = $env:DEEPSEEK_API_KEY
if (-not $apiKey) { $apiKey = Get-EnvFileValue 'DEEPSEEK_API_KEY' }
if (-not $apiKey) {
    $sameAsExample = (Get-FileHash '.env').Hash -eq (Get-FileHash '.env.example').Hash
    Write-Note 'DEEPSEEK_API_KEY is empty in .env - every analysis would fail.'
    if ($sameAsExample) {
        Write-Note '.env is byte-for-byte identical to .env.example: it was overwritten by the template.'
        Write-Note 'Your real key was lost with it - paste it back with: notepad .env'
    } else {
        Write-Note 'Set it with: notepad .env   (line: DEEPSEEK_API_KEY=sk-...)'
    }
    throw 'Refusing to start a demo without a model API key'
}
Write-Ok "DEEPSEEK_API_KEY present ($($apiKey.Length) chars)"

if (-not (Get-EnvFileValue 'LANGGRAPH_DEMO_API_TOKEN')) {
    Write-Note 'LANGGRAPH_DEMO_API_TOKEN is empty: the demo runs in open local mode.'
}

# --- 2) docker CLI present? -------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'docker CLI not found. Install Docker Desktop, then re-run this script.'
}

# --- 3) engine running? start Docker Desktop if needed ----------------------
function Test-DockerEngine {
    cmd /c "docker info >nul 2>&1" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-DockerEngine)) {
    Write-Note 'Docker engine is down - starting Docker Desktop...'
    $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path $desktop)) { throw "Docker Desktop not found at $desktop - start it manually." }
    Start-Process $desktop | Out-Null

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while (-not (Test-DockerEngine)) {
        if ((Get-Date) -ge $deadline) { throw "Docker engine did not become ready within $TimeoutSeconds s." }
        Start-Sleep -Seconds 5
        Write-Host '    ... still waiting for the engine'
    }
}
Write-Ok 'Docker engine is running'

# --- 4) RAG index present? --------------------------------------------------
$indexFiles = @(Get-ChildItem 'data/vectorstore' -Recurse -File -ErrorAction SilentlyContinue)
if ($indexFiles.Count -eq 0) {
    Write-Note 'No RAG index under data/vectorstore - answers will be degraded (no citations).'
    Write-Note 'Build it once: uv run python scripts/index_knowledge.py --knowledge-dir tests/fixtures/knowledge --rebuild'
} else {
    Write-Ok "RAG index present ($($indexFiles.Count) files)"
}

# --- 5) bring the stack up --------------------------------------------------
# Hugging Face offline mode: the embedding weights live in the named volume, so
# the hub metadata checks are pure waiting (each one fails and retries 5 times).
# Probe the volume and only stay online when the weights really are missing.
$hfVolume = 'psychegraph-hf-dev'
$hasWeights = $false
if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker run --rm -v "${hfVolume}:/cache/huggingface" psychegraph:dev `
        test -d /cache/huggingface/hub/models--BAAI--bge-m3 2>$null | Out-Null
    $hasWeights = ($LASTEXITCODE -eq 0)
}
if ($hasWeights) {
    $env:HF_HUB_OFFLINE = '1'
    Write-Ok 'Hugging Face weights cached: starting in offline mode (no hub retries)'
} else {
    $env:HF_HUB_OFFLINE = '0'
    Write-Note 'No cached embedding weights yet: this start downloads about 2.2 GB from huggingface.co.'
}

Write-Step 'Starting the stack (docker compose up -d)'
cmd /c "docker compose -f $composeFile --env-file .env up -d"
if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed - see the output above.' }
Write-Ok 'containers are up'

# --- 6) wait for /health ----------------------------------------------------
Write-Step "Waiting for $baseUrl/health"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$health = $null
while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 5
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
if ($null -eq $health) {
    throw "No answer from /health within $TimeoutSeconds s. Logs: docker compose -f $composeFile logs --tail 50"
}
Write-Ok ("health: status={0} graph_loaded={1} vectorstore_available={2}" -f $health.status, $health.graph_loaded, $health.vectorstore_available)
if ($health.status -ne 'ok') {
    Write-Note 'Service reports degraded - do not demo the citation panel until the index is mounted.'
}

# --- 7) prewarm so the first demo question is fast --------------------------
if (-not $SkipPrewarm) {
    Write-Step 'Prewarming the embedding model (about 30 s when the weights are cached)'
    cmd /c "docker compose -f $composeFile --env-file .env exec -T psychegraph python scripts/prewarm_rag.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Note 'Prewarm failed (not fatal) - the first question will just be slower.'
    } else {
        Write-Ok 'embedding model loaded'
    }
}

# --- 8) open the browser ----------------------------------------------------
if (-not $NoBrowser) {
    Start-Process $baseUrl | Out-Null
    Write-Ok "opened $baseUrl"
}
Write-Host ''
Write-Host "Demo ready: $baseUrl" -ForegroundColor Green
Write-Host "Stop it with demo_down.cmd when you are done." -ForegroundColor DarkGray
