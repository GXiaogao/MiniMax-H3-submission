param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\..\artifacts\infinifabric-toolchain-input')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$xmakeVersion = '3.0.5'
$xmakeSha256 = '4075ef1b4cba7f5eb5c55a4451761ef081086bb3142e7d4b130cf9c7484e68f0'
$boostVersion = '1.92.0'
$boostSha256 = 'f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984'
$pybindVersion = 'v3.1.0'
$pybindSha256 = 'affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536'
$xmakeRepoCommit = '2d3fb0bd267263ceaf4e527b3bc9fe3ce37fe11a'

$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-VerifiedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $existing = Get-Sha256 -Path $Destination
        if ($existing -eq $ExpectedSha256) {
            Write-Output "already verified: $Destination"
            return
        }
        Write-Output "replacing mismatched file: $Destination"
    }

    $temporary = $Destination + '.download'
    Write-Output "downloading: $Url"
    try {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $temporary
    }
    catch {
        Write-Output 'Invoke-WebRequest failed; retrying with curl'
        & curl.exe --fail --location --retry 2 --retry-all-errors --connect-timeout 30 `
            --continue-at - --output $temporary $Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed with exit code ${LASTEXITCODE}: $Url"
        }
    }

    $actual = Get-Sha256 -Path $temporary
    if ($actual -ne $ExpectedSha256) {
        Remove-Item -LiteralPath $temporary -Force
        throw "SHA256 mismatch for $Url; expected=$ExpectedSha256 actual=$actual"
    }
    Move-Item -LiteralPath $temporary -Destination $Destination -Force
    Write-Output "verified: $Destination"
}

$xmakePath = Join-Path $outputRoot 'xmake'
$boostPath = Join-Path $outputRoot 'boost-1.92.0-cmake.tar.gz'
$pybindPath = Join-Path $outputRoot 'pybind11-v3.1.0.zip'
$repoCachePath = Join-Path $outputRoot 'xmake-repo-cache.tar.gz'

Get-VerifiedFile `
    -Url "https://github.com/xmake-io/xmake/releases/download/v$xmakeVersion/xmake-bundle-v$xmakeVersion.linux.x86_64" `
    -Destination $xmakePath `
    -ExpectedSha256 $xmakeSha256

Get-VerifiedFile `
    -Url "https://github.com/boostorg/boost/releases/download/boost-$boostVersion/boost-$boostVersion-cmake.tar.gz" `
    -Destination $boostPath `
    -ExpectedSha256 $boostSha256

Get-VerifiedFile `
    -Url "https://github.com/pybind/pybind11/archive/refs/tags/$pybindVersion.zip" `
    -Destination $pybindPath `
    -ExpectedSha256 $pybindSha256

$xmakeBytes = [IO.File]::ReadAllBytes($xmakePath)
if ($xmakeBytes.Length -lt 4 -or
    $xmakeBytes[0] -ne 0x7f -or
    $xmakeBytes[1] -ne 0x45 -or
    $xmakeBytes[2] -ne 0x4c -or
    $xmakeBytes[3] -ne 0x46) {
    throw 'The downloaded xmake file is not an ELF executable.'
}

$stageRoot = Join-Path ([IO.Path]::GetTempPath()) ('infinifabric-xmake-repo-' + [guid]::NewGuid().ToString('N'))
$repoZip = Join-Path $stageRoot 'xmake-repo.zip'
try {
    New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
    $repoUrl = "https://codeload.github.com/xmake-io/xmake-repo/zip/$xmakeRepoCommit"
    Write-Output "downloading fixed xmake-repo commit: $xmakeRepoCommit"
    try {
        Remove-Item -LiteralPath $repoZip -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -UseBasicParsing -Uri $repoUrl -OutFile $repoZip
    }
    catch {
        Write-Output 'Invoke-WebRequest failed; retrying xmake-repo with curl'
        & curl.exe --fail --location --retry 2 --retry-all-errors --connect-timeout 30 `
            --output $repoZip $repoUrl
        if ($LASTEXITCODE -ne 0) {
            throw "xmake-repo download failed with exit code $LASTEXITCODE"
        }
    }

    $expanded = Join-Path $stageRoot 'expanded'
    Expand-Archive -LiteralPath $repoZip -DestinationPath $expanded
    $repoSource = Get-ChildItem -LiteralPath $expanded -Directory | Select-Object -First 1
    if ($null -eq $repoSource) {
        throw 'The xmake-repo archive did not contain a source directory.'
    }

    $boostRecipe = Join-Path $repoSource.FullName 'packages\b\boost\xmake.lua'
    $pybindRecipe = Join-Path $repoSource.FullName 'packages\p\pybind11\xmake.lua'
    if (-not (Test-Path -LiteralPath $boostRecipe -PathType Leaf) -or
        -not (Test-Path -LiteralPath $pybindRecipe -PathType Leaf)) {
        throw 'The fixed xmake-repo commit is missing required recipes.'
    }
    $boostRecipeText = Get-Content -LiteralPath $boostRecipe -Raw
    $pybindRecipeText = Get-Content -LiteralPath $pybindRecipe -Raw
    $expectedBoostRecipeLine = 'add_versions("cmake:' + $boostVersion + '", "' + $boostSha256 + '")'
    $expectedPybindRecipeLine = 'add_versions("' + $pybindVersion + '", "' + $pybindSha256 + '")'
    if (-not $boostRecipeText.Contains($expectedBoostRecipeLine)) {
        throw 'The Boost recipe does not contain the expected version and SHA256.'
    }
    if (-not $pybindRecipeText.Contains($expectedPybindRecipeLine)) {
        throw 'The pybind11 recipe does not contain the expected version and SHA256.'
    }

    $repoDestination = Join-Path $stageRoot '.xmake\repositories\xmake-repo'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $repoDestination) | Out-Null
    Move-Item -LiteralPath $repoSource.FullName -Destination $repoDestination

    Remove-Item -LiteralPath $repoCachePath -Force -ErrorAction SilentlyContinue
    & tar -czf $repoCachePath -C $stageRoot .xmake
    if ($LASTEXITCODE -ne 0) {
        throw 'tar failed while creating xmake-repo-cache.tar.gz'
    }
}
finally {
    if (Test-Path -LiteralPath $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}

$payloads = @(
    'xmake',
    'xmake-repo-cache.tar.gz',
    'boost-1.92.0-cmake.tar.gz',
    'pybind11-v3.1.0.zip'
)
$manifestLines = foreach ($name in $payloads) {
    $path = Join-Path $outputRoot $name
    "$(Get-Sha256 -Path $path)  $name"
}
[IO.File]::WriteAllText(
    (Join-Path $outputRoot 'SHA256SUMS'),
    (($manifestLines -join "`n") + "`n"),
    [Text.UTF8Encoding]::new($false)
)

$lock = [ordered]@{
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    xmake = [ordered]@{ version = $xmakeVersion; commit = 'f20295f255d967af9d91de6f1ed5eeadf17d30a7'; sha256 = $xmakeSha256 }
    xmake_repo = [ordered]@{ commit = $xmakeRepoCommit; archive_sha256 = (Get-Sha256 -Path $repoCachePath) }
    boost = [ordered]@{ version = $boostVersion; sha256 = $boostSha256 }
    pybind11 = [ordered]@{ version = $pybindVersion; sha256 = $pybindSha256 }
}
[IO.File]::WriteAllText(
    (Join-Path $outputRoot 'SOURCE_LOCK.json'),
    (($lock | ConvertTo-Json -Depth 4) + "`n"),
    [Text.UTF8Encoding]::new($false)
)

Write-Output "output=$outputRoot"
Get-ChildItem -LiteralPath $outputRoot -File |
    Sort-Object Name |
    Select-Object Name, Length, @{Name='SHA256'; Expression={ Get-Sha256 -Path $_.FullName }} |
    Format-Table -AutoSize
Write-Output 'InfiniFabric toolchain inputs: PASSED'
