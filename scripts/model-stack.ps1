param(
    [Parameter(Position = 0)]
    [string]$Action = "help",

    [Parameter(Position = 1)]
    [string]$Target = "all",

    [Parameter(Position = 2)]
    [int]$Lines = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root ".run"
$LogDir = Join-Path $Root "logs"
$EnvFile = if ($env:MODEL_ENV_FILE) { $env:MODEL_ENV_FILE } else { Join-Path $Root ".env.runtime" }

$Ports = @{
    "layout" = 8001
    "detection" = 8002
    "rec-th" = 8004
    "ocr-custom" = 8005
    "ocr-paddle" = 8006
    "table-wired" = 8007
    "table-wireless" = 8008
    "siglip" = 8009
    "layout-pipeline" = 8010
    "table-pipeline" = 8011
    "image-verification" = 8012
    "table-v2" = 8013
    "gateway" = 8080
    "demo" = 8501
}
$PortEnvironment = @{
    "layout" = "LAYOUT_PORT"
    "detection" = "DETECTION_PORT"
    "rec-th" = "REC_TH_PORT"
    "ocr-custom" = "OCR_CUSTOM_PORT"
    "ocr-paddle" = "OCR_PADDLE_PORT"
    "table-wired" = "TABLE_WIRED_PORT"
    "table-wireless" = "TABLE_WIRELESS_PORT"
    "siglip" = "SIGLIP_PORT"
    "layout-pipeline" = "LAYOUT_PIPELINE_PORT"
    "table-pipeline" = "TABLE_PIPELINE_PORT"
    "image-verification" = "IMAGE_VERIFICATION_PORT"
    "table-v2" = "TABLE_V2_PORT"
    "gateway" = "GATEWAY_PORT"
    "demo" = "DEMO_PORT"
}

$CoreServices = @(
    "layout", "detection", "rec-th", "table-wired", "table-wireless", "siglip",
    "ocr-custom", "layout-pipeline", "table-pipeline", "image-verification", "gateway"
)
$AllServices = @(
    "layout", "detection", "rec-th", "table-wired", "table-wireless", "table-v2", "siglip",
    "ocr-custom", "ocr-paddle", "layout-pipeline", "table-pipeline", "image-verification", "gateway"
)
$Profiles = @{
    "core-stack" = $CoreServices
    "ocr-custom-stack" = @("detection", "rec-th", "ocr-custom", "gateway")
    "ocr-paddle-stack" = @("ocr-paddle", "gateway")
    "layout-stack" = @("layout", "detection", "layout-pipeline", "gateway")
    "table-stack" = @("detection", "rec-th", "ocr-custom", "table-wired", "table-wireless", "table-pipeline", "gateway")
    "table-v2-stack" = @("table-v2", "gateway")
    "verification-stack" = @("siglip", "image-verification", "gateway")
}
$ProfilePipelines = @{
    "core-stack" = "layout,ocr-custom,table,image-verification,text-detection,text-recognition,siglip"
    "ocr-custom-stack" = "ocr-custom,text-detection,text-recognition"
    "ocr-paddle-stack" = "ocr-paddle"
    "layout-stack" = "layout,text-detection"
    "table-stack" = "table,text-detection,text-recognition"
    "table-v2-stack" = "table-model"
    "verification-stack" = "image-verification,siglip"
    "all" = "all"
}

function Show-Usage {
    @"
Usage:
  scripts\model-stack.cmd start <service|profile>
  scripts\model-stack.cmd stop <service|profile|all>
  scripts\model-stack.cmd restart <service|profile>
  scripts\model-stack.cmd status [service|profile|all]
  scripts\model-stack.cmd readiness [service|profile|all]
  scripts\model-stack.cmd logs <service> [lines]
  scripts\model-stack.cmd check

Profiles:
  core-stack          default production stack using split polygon det + rec OCR
  ocr-custom-stack    detection + rec-th + custom OCR + gateway
  ocr-paddle-stack    integrated Paddle OCR + gateway
  layout-stack        layout + detection + layout pipeline + gateway
  table-stack         split wired/wireless table pipeline + gateway
  table-v2-stack      TableRecognitionPipelineV2 + gateway; no split table models
  verification-stack SigLIP + verification pipeline + gateway
  all                 every model/pipeline; may load duplicate weights

Services:
  layout detection rec-th ocr-custom ocr-paddle table-wired
  table-wireless table-v2 siglip layout-pipeline table-pipeline
  image-verification gateway demo
"@ | Write-Host
}

function Import-RuntimeEnvironment {
    if (-not (Test-Path -LiteralPath $EnvFile)) {
        return
    }
    foreach ($rawLine in Get-Content -LiteralPath $EnvFile) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }
        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

function Get-Targets([string]$Name, [bool]$ForStart = $false) {
    $normalized = $Name.Trim().ToLowerInvariant()
    if ($normalized -eq "all") {
        if ($ForStart) { return $AllServices }
        return @($AllServices + "demo")
    }
    if ($Profiles.ContainsKey($normalized)) {
        return @($Profiles[$normalized])
    }
    if ($Ports.ContainsKey($normalized)) {
        return @($normalized)
    }
    throw "Unknown service/profile: $Name"
}

function Get-ProcessIdFile([string]$Service) {
    return Join-Path $RunDir "$Service.windows.pid"
}

function Get-OutputLog([string]$Service) {
    return Join-Path $LogDir "$Service.windows.log"
}

function Get-ErrorLog([string]$Service) {
    return Join-Path $LogDir "$Service.windows.error.log"
}

function Get-TrackedProcess([string]$Service) {
    $pidFile = Get-ProcessIdFile $Service
    if (-not (Test-Path -LiteralPath $pidFile)) {
        return $null
    }
    $rawPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    $processId = 0
    if (-not [int]::TryParse($rawPid, [ref]$processId)) {
        Remove-Item -LiteralPath $pidFile -Force
        return $null
    }
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        $details = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction Stop
        $expectedRunner = if ($Service -eq "demo") { "run-demo.cmd" } else { "run-service.cmd" }
        if (-not $details.CommandLine -or $details.CommandLine -notlike "*$expectedRunner*") {
            Remove-Item -LiteralPath $pidFile -Force
            return $null
        }
        if ($Service -ne "demo" -and $details.CommandLine -notlike "*$Service*") {
            Remove-Item -LiteralPath $pidFile -Force
            return $null
        }
        return $process
    }
    catch {
        Remove-Item -LiteralPath $pidFile -Force
        return $null
    }
}

function Test-PortInUse([int]$Port) {
    try {
        return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop | Select-Object -First 1)
    }
    catch {
        return $false
    }
}

function Get-EnvironmentInteger([string]$Name, [int]$DefaultValue) {
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    $parsed = 0
    if ($value -and [int]::TryParse($value, [ref]$parsed) -and $parsed -gt 0) {
        return $parsed
    }
    return $DefaultValue
}

function Wait-ServiceLive([string]$Service, [System.Diagnostics.Process]$Process) {
    $port = [int]$Ports[$Service]
    $path = if ($Service -eq "demo") { "/" } else { "/health" }
    $deadline = (Get-Date).AddSeconds((Get-EnvironmentInteger "STARTUP_WAIT_SECONDS" 45))
    while ((Get-Date) -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            return $false
        }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port$path" -TimeoutSec 2 | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Start-ManagedService([string]$Service, [string]$GatewayPipelines) {
    $existing = Get-TrackedProcess $Service
    if ($null -ne $existing) {
        Write-Host "[SKIP] $Service is already running (PID $($existing.Id))."
        return
    }
    $port = [int]$Ports[$Service]
    if (Test-PortInUse $port) {
        throw "Port $port for $Service is already in use by an unmanaged process."
    }

    $runner = if ($Service -eq "demo") {
        Join-Path $PSScriptRoot "run-demo.cmd"
    }
    else {
        Join-Path $PSScriptRoot "run-service.cmd"
    }
    $cmdCommand = if ($Service -eq "demo") {
        "`"`"$runner`"`""
    }
    else {
        "`"`"$runner`" `"$Service`"`""
    }
    $oldGatewayPipelines = [Environment]::GetEnvironmentVariable("GATEWAY_ENABLED_PIPELINES", "Process")
    if ($Service -eq "gateway" -and $GatewayPipelines) {
        [Environment]::SetEnvironmentVariable("GATEWAY_ENABLED_PIPELINES", $GatewayPipelines, "Process")
    }
    try {
        Write-Host "[START] $Service on 127.0.0.1:$port"
        $startOptions = @{
            FilePath = $env:ComSpec
            ArgumentList = @("/d", "/s", "/c", $cmdCommand)
            WorkingDirectory = $Root
            WindowStyle = "Hidden"
            RedirectStandardOutput = Get-OutputLog $Service
            RedirectStandardError = Get-ErrorLog $Service
            PassThru = $true
        }
        $process = Start-Process @startOptions
    }
    finally {
        if ($Service -eq "gateway" -and $GatewayPipelines) {
            [Environment]::SetEnvironmentVariable("GATEWAY_ENABLED_PIPELINES", $oldGatewayPipelines, "Process")
        }
    }
    Set-Content -LiteralPath (Get-ProcessIdFile $Service) -Value $process.Id -Encoding ascii
    if (Wait-ServiceLive $Service $process) {
        Write-Host "[OK] $Service is live (PID $($process.Id))."
        return
    }
    if ($process.HasExited) {
        Remove-Item -LiteralPath (Get-ProcessIdFile $Service) -Force -ErrorAction SilentlyContinue
    }
    Write-Host "[ERROR] $Service did not become live. Logs:" -ForegroundColor Red
    Write-Host "        $(Get-OutputLog $Service)"
    Write-Host "        $(Get-ErrorLog $Service)"
    throw "$Service startup failed."
}

function Stop-ManagedService([string]$Service) {
    $process = Get-TrackedProcess $Service
    if ($null -eq $process) {
        Write-Host "[SKIP] $Service is not managed or already stopped."
        return
    }
    Write-Host "[STOP] $Service (PID $($process.Id))"
    & taskkill.exe /PID $process.Id /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not stop $Service PID $($process.Id)."
    }
    Remove-Item -LiteralPath (Get-ProcessIdFile $Service) -Force -ErrorAction SilentlyContinue
    Write-Host "[OK] $Service stopped."
}

function Show-ServiceStatus([string]$Service) {
    $process = Get-TrackedProcess $Service
    $port = [int]$Ports[$Service]
    if ($null -ne $process) {
        Write-Host ("{0,-24} port={1,-5} pid={2,-8} state=running" -f $Service, $port, $process.Id)
    }
    elseif (Test-PortInUse $port) {
        Write-Host ("{0,-24} port={1,-5} pid={2,-8} state=unmanaged" -f $Service, $port, "-")
    }
    else {
        Write-Host ("{0,-24} port={1,-5} pid={2,-8} state=stopped" -f $Service, $port, "-")
    }
}

function Show-ServiceReadiness([string]$Service) {
    if ($Service -eq "demo") {
        Show-ServiceStatus $Service
        return
    }
    $port = [int]$Ports[$Service]
    $tokenName = if ($Service -eq "gateway") { "MODEL_GATEWAY_API_KEY" } else { "INTERNAL_API_TOKEN" }
    $token = [Environment]::GetEnvironmentVariable($tokenName, "Process")
    $headers = @{}
    if ($token) { $headers["Authorization"] = "Bearer $token" }
    Write-Host "--- $Service :$port readiness ---"
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/v1/readiness" -Headers $headers -TimeoutSec (Get-EnvironmentInteger "READINESS_TIMEOUT_SECONDS" 300)
        Write-Host $response.Content
        Write-Host "HTTP $($response.StatusCode)"
    }
    catch {
        if ($_.Exception.Response) {
            Write-Host "HTTP $([int]$_.Exception.Response.StatusCode)"
        }
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

function Show-Check {
    $requirements = @(
        @{ Name = "API"; Path = Join-Path $Root ".venv-api312\Scripts\python.exe" },
        @{ Name = "Paddle"; Path = Join-Path $Root ".venv-paddle312\Scripts\python.exe" },
        @{ Name = "SigLIP"; Path = Join-Path $Root ".venv-siglip312\Scripts\python.exe" }
    )
    foreach ($requirement in $requirements) {
        if (Test-Path -LiteralPath $requirement.Path) {
            $version = & $requirement.Path --version 2>&1
            Write-Host "[OK] $($requirement.Name): $version"
        }
        else {
            Write-Host "[MISSING] $($requirement.Name): $($requirement.Path)" -ForegroundColor Yellow
        }
    }
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        Write-Host "[OK] nvidia-smi is available."
    }
    else {
        Write-Host "[WARN] nvidia-smi is unavailable; gpu:0 services cannot start." -ForegroundColor Yellow
    }
}

New-Item -ItemType Directory -Path $RunDir, $LogDir -Force | Out-Null
Import-RuntimeEnvironment
foreach ($serviceName in $PortEnvironment.Keys) {
    $configuredPort = [Environment]::GetEnvironmentVariable($PortEnvironment[$serviceName], "Process")
    $parsedPort = 0
    if ($configuredPort -and [int]::TryParse($configuredPort, [ref]$parsedPort) -and $parsedPort -ge 1 -and $parsedPort -le 65535) {
        $Ports[$serviceName] = $parsedPort
    }
}
$Action = $Action.Trim().ToLowerInvariant()
$Target = $Target.Trim().ToLowerInvariant()

try {
    switch ($Action) {
        "start" {
            $targets = @(Get-Targets $Target $true)
            $gatewayPipelines = if ($ProfilePipelines.ContainsKey($Target)) { $ProfilePipelines[$Target] } else { "" }
            foreach ($service in $targets) { Start-ManagedService $service $gatewayPipelines }
        }
        "stop" {
            $targets = @(Get-Targets $Target $false)
            [array]::Reverse($targets)
            foreach ($service in $targets) { Stop-ManagedService $service }
        }
        "restart" {
            $targets = @(Get-Targets $Target $false)
            $reversed = @($targets)
            [array]::Reverse($reversed)
            foreach ($service in $reversed) { Stop-ManagedService $service }
            $gatewayPipelines = if ($ProfilePipelines.ContainsKey($Target)) { $ProfilePipelines[$Target] } else { "" }
            foreach ($service in @(Get-Targets $Target $true)) { Start-ManagedService $service $gatewayPipelines }
        }
        "status" {
            foreach ($service in @(Get-Targets $Target $false)) { Show-ServiceStatus $service }
        }
        "readiness" {
            foreach ($service in @(Get-Targets $Target $false)) { Show-ServiceReadiness $service }
        }
        "logs" {
            $services = @(Get-Targets $Target $false)
            if ($services.Count -ne 1) { throw "logs requires one service name, not a profile." }
            $paths = @((Get-OutputLog $services[0]), (Get-ErrorLog $services[0])) | Where-Object { Test-Path -LiteralPath $_ }
            if (-not $paths) { throw "No Windows log exists for $($services[0])." }
            Get-Content -LiteralPath $paths -Tail ([Math]::Max(1, $Lines)) -Wait
        }
        "check" { Show-Check }
        "help" { Show-Usage }
        "-h" { Show-Usage }
        "--help" { Show-Usage }
        default { Show-Usage; exit 2 }
    }
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
