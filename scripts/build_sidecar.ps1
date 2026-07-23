$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BuildRoot = Join-Path $ProjectRoot "build"
$OutputRoot = Join-Path $BuildRoot "sidecar"
$Target = Join-Path $OutputRoot "run_omni_providers.dist"
$StagingRoot = Join-Path $BuildRoot "pyinstaller_dist"
$StagingTarget = Join-Path $StagingRoot "run_omni_providers"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python do ambiente virtual nao encontrado em: $Python. Execute scripts/setup.ps1 primeiro."
}

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    $ResolvedBuildRoot = (Resolve-Path -LiteralPath $BuildRoot).Path.TrimEnd("\") + "\"
    $ResolvedTargetPrefix = [System.IO.Path]::GetFullPath($Target).TrimEnd("\") + "\"
    $RunningFromOutput = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and $_.Path.StartsWith($ResolvedTargetPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    }
    if ($RunningFromOutput) {
        throw "Pare o OmniProviders empacotado antes de iniciar outro build."
    }

    foreach ($PathToClean in @($Target, $StagingRoot, (Join-Path $BuildRoot "pyinstaller_work"), (Join-Path $BuildRoot "pyinstaller_spec"))) {
        if (Test-Path -LiteralPath $PathToClean) {
            $ResolvedTarget = (Resolve-Path -LiteralPath $PathToClean).Path
            if (-not $ResolvedTarget.StartsWith($ResolvedBuildRoot)) {
                throw "Caminho de build recusado para limpeza: $ResolvedTarget"
            }
            Remove-Item -LiteralPath $ResolvedTarget -Recurse -Force
        }
    }

    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --name run_omni_providers `
        --distpath $StagingRoot `
        --workpath (Join-Path $BuildRoot "pyinstaller_work") `
        --specpath (Join-Path $BuildRoot "pyinstaller_spec") `
        --collect-all google.genai `
        --collect-all gemini_webapi `
        --collect-all curl_cffi `
        --collect-all playwright `
        run_omni_providers.py

    if (-not (Test-Path -LiteralPath (Join-Path $StagingTarget "run_omni_providers.exe"))) {
        throw "PyInstaller nao produziu o executavel esperado."
    }
    Move-Item -LiteralPath $StagingTarget -Destination $Target

    $PlaywrightNodeCandidates = @(
        (Join-Path $Target "playwright\driver\node.exe"),
        (Join-Path $Target "_internal\playwright\driver\node.exe")
    )
    $ResolvedTargetRoot = (Resolve-Path -LiteralPath $Target).Path.TrimEnd("\") + "\"
    foreach ($PlaywrightNode in $PlaywrightNodeCandidates) {
        if (-not (Test-Path -LiteralPath $PlaywrightNode)) {
            continue
        }
        $ResolvedNode = (Resolve-Path -LiteralPath $PlaywrightNode).Path
        if (-not $ResolvedNode.StartsWith($ResolvedTargetRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Caminho de node.exe recusado para limpeza: $ResolvedNode"
        }
        Remove-Item -LiteralPath $ResolvedNode -Force
    }

    foreach ($GeneratedPath in @($StagingRoot, (Join-Path $BuildRoot "pyinstaller_work"), (Join-Path $BuildRoot "pyinstaller_spec"))) {
        if (Test-Path -LiteralPath $GeneratedPath) {
            $ResolvedGeneratedPath = (Resolve-Path -LiteralPath $GeneratedPath).Path
            if (-not $ResolvedGeneratedPath.StartsWith($ResolvedBuildRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Caminho gerado recusado para limpeza: $ResolvedGeneratedPath"
            }
            Remove-Item -LiteralPath $ResolvedGeneratedPath -Recurse -Force
        }
    }
}
finally {
    Pop-Location
}
