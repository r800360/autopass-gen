# Production showcase batch — one MP4 per scenario family (real LLM + CARLA).
# Usage (from repo root, CARLA running):
#   .\.venv\Scripts\activate
#   $env:OPENAI_API_KEY = "<your-key>"
#   .\scripts\run_showcase_batch.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:AUTOPASS_TEST_MODE = "0"
$env:AUTOPASS_MOCK_LLM = "0"
$env:AUTOPASS_DECISION_ORACLE = "0"
$env:AUTOPASS_LLM_TEMPERATURE = "0.4"
$env:AUTOPASS_EXECUTE_DT_S = "1.0"
$env:AUTOPASS_VIDEO_REALTIME = "1"
$env:AUTOPASS_DEMO_DENSE_FRAMES = "1"

$runs = @(
    @{ Out = "runs/showcase_clear_safe_pass"; Args = @("--hero-pass", "--scenario", "clear_safe_pass", "--urgency", "high", "--steps", "60", "--ticks", "10") },
    @{ Out = "runs/showcase_slow_lead_high"; Args = @("--hero-pass", "--scenario", "slow_lead_high_urgency", "--urgency", "high", "--steps", "55", "--ticks", "10") },
    @{ Out = "runs/showcase_demo07"; Args = @("--hero-pass", "--scenario", "6", "--urgency", "high", "--steps", "60", "--ticks", "10") }
)

foreach ($r in $runs) {
    $out = $r.Out
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "`n=== $out ===" -ForegroundColor Cyan
    python demo_carla_watch.py @($r.Args) --out-dir $out
    if ($LASTEXITCODE -ne 0) { throw "demo failed: $out" }
}

Write-Host "`nDone. Inspect runs/showcase_*/frames and *.mp4" -ForegroundColor Green
