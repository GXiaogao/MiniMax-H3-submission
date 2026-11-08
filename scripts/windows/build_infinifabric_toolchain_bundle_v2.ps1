param(
    [Parameter(Mandatory = $true)]
    [string]$InputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'

$inputRoot = (Resolve-Path -LiteralPath $InputDirectory).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$outputRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path

$requiredFiles = @(
    'xmake',
    'xmake-repo-cache.tar.gz',
    'boost-1.92.0-cmake.tar.xz',
    'pybind11-v3.1.0.zip'
)
foreach ($name in $requiredFiles) {
    $path = Join-Path $inputRoot $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input is missing: $path"
    }
}

$repoCache = Join-Path $inputRoot 'xmake-repo-cache.tar.gz'
$repoEntries = & tar -tzf $repoCache
if ($LASTEXITCODE -ne 0) {
    throw "Cannot read xmake repository cache: $repoCache"
}
$normalizedEntries = $repoEntries | ForEach-Object { $_ -replace '\\', '/' }
$requiredRecipes = @(
    '.xmake/repositories/xmake-repo/packages/b/boost/xmake.lua',
    '.xmake/repositories/xmake-repo/packages/p/pybind11/xmake.lua'
)
foreach ($recipe in $requiredRecipes) {
    if ($normalizedEntries -notcontains $recipe) {
        throw "Repository cache is missing required recipe: $recipe"
    }
}

$bundleName = 'infinifabric-toolchain-v2'
$stageParent = Join-Path $outputRoot ('.stage-' + [guid]::NewGuid().ToString('N'))
$stageRoot = Join-Path $stageParent $bundleName
$packageRoot = Join-Path $stageRoot 'packages'
$archivePath = Join-Path $outputRoot ($bundleName + '.tar.gz')
$archiveHashPath = $archivePath + '.sha256'

try {
    New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $inputRoot 'xmake') -Destination (Join-Path $stageRoot 'xmake')
    Copy-Item -LiteralPath $repoCache -Destination (Join-Path $packageRoot 'xmake-repo-cache.tar.gz')
    Copy-Item -LiteralPath (Join-Path $inputRoot 'boost-1.92.0-cmake.tar.xz') -Destination $packageRoot
    Copy-Item -LiteralPath (Join-Path $inputRoot 'pybind11-v3.1.0.zip') -Destination $packageRoot

    $payloads = @(
        @{ Relative = 'xmake'; Path = (Join-Path $stageRoot 'xmake') },
        @{ Relative = 'packages/boost-1.92.0-cmake.tar.xz'; Path = (Join-Path $packageRoot 'boost-1.92.0-cmake.tar.xz') },
        @{ Relative = 'packages/pybind11-v3.1.0.zip'; Path = (Join-Path $packageRoot 'pybind11-v3.1.0.zip') },
        @{ Relative = 'packages/xmake-repo-cache.tar.gz'; Path = (Join-Path $packageRoot 'xmake-repo-cache.tar.gz') }
    )
    $manifestLines = foreach ($payload in $payloads) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $payload.Path).Hash.ToLowerInvariant()
        "$hash  $($payload.Relative)"
    }
    $manifestText = ($manifestLines -join "`n") + "`n"
    [IO.File]::WriteAllText(
        (Join-Path $stageRoot 'SHA256SUMS'),
        $manifestText,
        [Text.UTF8Encoding]::new($false)
    )

    Push-Location $stageParent
    try {
        & tar -czf $archivePath $bundleName
        if ($LASTEXITCODE -ne 0) {
            throw 'tar failed while creating the toolchain bundle'
        }
    }
    finally {
        Pop-Location
    }

    $archive = Get-Item -LiteralPath $archivePath
    if ($archive.Length -ge 128MB) {
        throw "Bundle exceeds the InfiniFabric 128 MiB single-file limit: $($archive.Length) bytes"
    }

    $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText(
        $archiveHashPath,
        "$archiveHash  $($archive.Name)`n",
        [Text.UTF8Encoding]::new($false)
    )

    Write-Output "bundle=$archivePath"
    Write-Output "sha256=$archiveHash"
    Write-Output "size_bytes=$($archive.Length)"
    Write-Output 'toolchain bundle: PASSED'
}
finally {
    $resolvedStage = [IO.Path]::GetFullPath($stageParent)
    $resolvedOutput = [IO.Path]::GetFullPath($outputRoot).TrimEnd('\') + '\'
    if ($resolvedStage.StartsWith($resolvedOutput, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedStage)) {
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}
