$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    throw '请先运行 uv sync --locked，再执行启动脚本。'
}
& .venv/Scripts/python.exe -m annual_report_agent start
exit $LASTEXITCODE
