$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    throw 'Activate the target Conda environment first, for example: conda activate dino_VLM'
}

$environmentName = Split-Path -Leaf $env:CONDA_PREFIX
if ($environmentName -eq 'Anaconda3') {
    throw 'The base environment is active. Run conda activate dino_VLM in this PowerShell window before installing tooth_vlm.'
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$target = Join-Path $env:CONDA_PREFIX 'Scripts\tooth_vlm.cmd'
$expectedPython = Join-Path $env:CONDA_PREFIX 'python.exe'
$python = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source

if ((Resolve-Path $python).Path -ne (Resolve-Path $expectedPython).Path) {
    throw "The active Conda prefix and python command disagree. CONDA_PREFIX=$env:CONDA_PREFIX; python=$python"
}

@"
@echo off
"$python" "$projectRoot\app\scripts\launch.py"
exit /b %errorlevel%
"@ | Set-Content -LiteralPath $target -Encoding ascii

Write-Host "Installed tooth_vlm in the active Conda environment: $env:CONDA_PREFIX"
Write-Host 'After manually running conda activate dino_VLM, start the app with: tooth_vlm'
