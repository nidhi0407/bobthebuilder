# PowerShell Quick Trigger Script for BobTheBuilder Demo
param(
    [string]$Worker = "Warning worker! Forklift approaching blind corner on your right!",
    [string]$Forklift = "Emergency stop! Worker detected in vehicle corridor 4 meters ahead!"
)

$payload = @{
    worker_message = $Worker
    forklift_message = $Forklift
} | ConvertTo-Json

Write-Host "🚨 Triggering ForeSite Emergency Alert..." -ForegroundColor Yellow
$response = Invoke-RestMethod -Uri "http://localhost:5000/trigger" -Method Post -ContentType "application/json" -Body $payload
Write-Host "Alert State:" -ForegroundColor Cyan
$response.state | Format-List
