# PowerShell Quick Clear Script for BobTheBuilder Demo
Write-Host "🟢 Sending All Clear Signal..." -ForegroundColor Green
$response = Invoke-RestMethod -Uri "http://localhost:5000/clear" -Method Post
$response.state | Format-List
