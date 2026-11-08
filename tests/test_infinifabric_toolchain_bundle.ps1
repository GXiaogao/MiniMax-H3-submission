$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$packager = Join-Path $repoRoot 'scripts\windows\build_infinifabric_toolchain_bundle.ps1'
if (-not (Test-Path -LiteralPath $packager -PathType Leaf)) {
    throw "Packager is missing: $packager"
}

$scratch = Join-Path $env:TEMP ("infinifabric-toolchain-test-" + [guid]::NewGuid().ToString('N'))
$inputDir = Join-Path $scratch 'input'
$outputDir = Join-Path $scratch 'output'
$repoStage = Join-Path $scratch 'repo-stage\.xmake\repositories\xmake-repo'

try {
    New-Item -ItemType Directory -Force -Path $inputDir, $outputDir | Out-Null
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $repoStage 'packages\b\boost'), `
        (Join-Path $repoStage 'packages\p\pybind11') | Out-Null

    [IO.File]::WriteAllText((Join-Path $inputDir 'xmake'), "fixture-xmake`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $inputDir 'boost-1.92.0-cmake.tar.gz'), "boost`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $inputDir 'pybind11-v3.1.0.zip'), "pybind11`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $repoStage 'packages\b\boost\xmake.lua'), "boost recipe`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $repoStage 'packages\p\pybind11\xmake.lua'), "pybind recipe`n", [Text.UTF8Encoding]::new($false))

    Push-Location (Join-Path $scratch 'repo-stage')
    try {
        & tar -czf (Join-Path $inputDir 'xmake-repo-cache.tar.gz') '.xmake'
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create repository fixture' }
    }
    finally { Pop-Location }

    & $packager -InputDirectory $inputDir -OutputDirectory $outputDir
    if ($LASTEXITCODE -ne 0) { throw "Packager failed with exit code $LASTEXITCODE" }

    $archive = Join-Path $outputDir 'infinifabric-toolchain-v1.tar.gz'
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw 'Expected toolchain archive was not created'
    }

    $extractDir = Join-Path $scratch 'extract'
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    & tar -xzf $archive -C $extractDir
    if ($LASTEXITCODE -ne 0) { throw 'Generated archive cannot be extracted' }

    $root = Join-Path $extractDir 'infinifabric-toolchain-v1'
    $required = @(
        'xmake',
        'packages\boost-1.92.0-cmake.tar.gz',
        'packages\pybind11-v3.1.0.zip',
        'packages\xmake-repo-cache.tar.gz',
        'SHA256SUMS'
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
            throw "Generated archive is missing: $relative"
        }
    }

    $manifestBytes = [IO.File]::ReadAllBytes((Join-Path $root 'SHA256SUMS'))
    if (($manifestBytes | Where-Object { $_ -eq 13 }).Count -ne 0) {
        throw 'SHA256SUMS contains CRLF characters'
    }

    Write-Output 'Offline toolchain bundle fixture: PASSED'
}
finally {
    if (Test-Path -LiteralPath $scratch) {
        Remove-Item -LiteralPath $scratch -Recurse -Force
    }
}
