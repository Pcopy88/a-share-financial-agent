$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'
$taskTemp = 'tmp/pytest-' + [guid]::NewGuid().ToString('N')
& .venv/Scripts/python.exe -m pytest --basetemp=$taskTemp @args
exit $LASTEXITCODE
