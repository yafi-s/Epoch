param(
    [int]$LocalPort = 50051,
    [string]$NgrokPath = "tools/ngrok/ngrok.exe",
    [string]$Authtoken = "",
    [switch]$NoHold
)

$ErrorActionPreference = "Stop"

function Resolve-PublicAddress([string]$Url) {
    if ($Url -match "^tcp://(.+)$") {
        return $Matches[1]
    }
    throw "Unexpected ngrok URL format: $Url"
}

if (-not (Test-Path $NgrokPath)) {
    throw "ngrok binary not found at '$NgrokPath'."
}

if ($Authtoken) {
    & $NgrokPath config add-authtoken $Authtoken | Out-Host
}

$logDir = "results/ngrok"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdoutLog = Join-Path $logDir "ngrok.out.log"
$stderrLog = Join-Path $logDir "ngrok.err.log"

if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force }
if (Test-Path $stderrLog) { Remove-Item $stderrLog -Force }

$args = "tcp $LocalPort --log=stdout"
$proc = Start-Process -FilePath $NgrokPath -ArgumentList $args -PassThru `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

try {
    $publicAddress = $null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { break }

        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
            $tcpTunnel = @($resp.tunnels | Where-Object { $_.proto -eq "tcp" }) | Select-Object -First 1
            if ($tcpTunnel -and $tcpTunnel.public_url) {
                $publicAddress = Resolve-PublicAddress $tcpTunnel.public_url
                break
            }
        } catch {
            # ngrok API not ready yet
        }
    }

    if (-not $publicAddress) {
        Write-Host "ngrok did not provide a public TCP tunnel."
        if (Test-Path $stdoutLog) {
            Write-Host ""
            Write-Host "---- ngrok stdout ----"
            Get-Content $stdoutLog | Select-Object -Last 50 | Out-Host
        }
        if (Test-Path $stderrLog) {
            Write-Host ""
            Write-Host "---- ngrok stderr ----"
            Get-Content $stderrLog | Select-Object -Last 50 | Out-Host
        }
        throw "Failed to create ngrok TCP tunnel."
    }

    Write-Host ""
    Write-Host "Scheduler public address: $publicAddress"
    Write-Host ""
    Write-Host "Set this before launching Modal workers:"
    Write-Host "  `$env:SCHEDULER_PUBLIC_ADDRESS = `"$publicAddress`""
    Write-Host ""
    Write-Host "Then run:"
    Write-Host "  modal run scripts/modal_workers.py --scheduler-address `"$publicAddress`" --gpu-type T4 --num-workers 8 --modal-secret-name superkey"

    if ($NoHold) {
        return
    }

    Write-Host ""
    Write-Host "ngrok tunnel running (PID: $($proc.Id)). Press Ctrl+C to stop."
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    if (-not $NoHold -and $proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
    }
}

