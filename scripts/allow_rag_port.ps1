# 放行 RAG 微服务端口 8090（局域网内同伴的 MacBook 需要访问）
# 用法：右键本文件 → “使用 PowerShell 运行”，或以管理员身份执行：
#   powershell -ExecutionPolicy Bypass -File scripts\allow_rag_port.ps1

$ruleName = "TCM RAG 8090"

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "请以管理员身份运行（右键 PowerShell → 以管理员身份运行）。" -ForegroundColor Yellow
    exit 1
}

$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "规则已存在，无需重复添加：$ruleName" -ForegroundColor Cyan
} else {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound `
        -LocalPort 8090 -Protocol TCP -Action Allow -Profile Any | Out-Null
    Write-Host "已添加入站放行规则：$ruleName (TCP 8090)" -ForegroundColor Green
}

Get-NetFirewallRule -DisplayName $ruleName | Format-Table DisplayName, Enabled, Direction, Action -AutoSize
Write-Host ""
Write-Host "同伴测试命令（在他的 MacBook 上执行）：" -ForegroundColor Cyan
Write-Host "  curl http://<本机热点IP>:8090/health"
