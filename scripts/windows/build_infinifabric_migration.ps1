param(
    [string]$ProjectRoot = "E:\projects\Codex_projects\MiniMax-H3"
)

$ErrorActionPreference = "Stop"

$reposRoot = Join-Path $ProjectRoot "repos"
$worktreesRoot = Join-Path $ProjectRoot "worktrees"
$outputRoot = Join-Path $ProjectRoot "artifacts\infinifabric-migration-v2"
$stageRoot = Join-Path $outputRoot "stage"
$archivePath = Join-Path $outputRoot "infinifabric-source-v2.tar.gz"
$checksumPath = "${archivePath}.sha256"

if (Test-Path -LiteralPath $outputRoot) {
    throw "Output already exists: $outputRoot"
}

New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot "Projects") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot "scripts") -Force | Out-Null

function Copy-SourceTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    & robocopy $Source $Destination /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
        /XD .git .xmake build __pycache__ .pytest_cache .mypy_cache `
        /XF .git "*.pyc" "*.pyo"

    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code ${LASTEXITCODE}: $Source"
    }
}

# Start from complete checkouts so all populated submodule source is retained.
Copy-SourceTree `
    -Source (Join-Path $reposRoot "InfiniCore") `
    -Destination (Join-Path $stageRoot "Projects\InfiniCore")

Copy-SourceTree `
    -Source (Join-Path $reposRoot "InfiniLM") `
    -Destination (Join-Path $stageRoot "Projects\InfiniLM")

# Overlay the validated competition changes from the dedicated worktrees.
Copy-SourceTree `
    -Source (Join-Path $worktreesRoot "InfiniCore-h3-mha") `
    -Destination (Join-Path $stageRoot "Projects\InfiniCore")

Copy-SourceTree `
    -Source (Join-Path $worktreesRoot "InfiniLM-qwen-video") `
    -Destination (Join-Path $stageRoot "Projects\InfiniLM")

# Include validation and adaptation scripts, excluding generated caches.
Copy-SourceTree `
    -Source (Join-Path $ProjectRoot "scripts") `
    -Destination (Join-Path $stageRoot "scripts")

$manifest = [ordered]@{
    format = 1
    created_at = (Get-Date).ToString("o")
    target = "InfiniFabric 8-card / DTK 26.04"
    excludes = @(".git", ".xmake", "build", "__pycache__", ".pytest_cache", ".mypy_cache", "*.pyc", "*.pyo")
    sources = [ordered]@{
        InfiniCore = [ordered]@{
            commit = "f32e2fbb2a9b2ca912b5711f4f3836c403196a74"
            branch = "minimax-h3"
            overlay = "worktrees/InfiniCore-h3-mha"
        }
        InfiniLM = [ordered]@{
            commit = "65deb91f840ebaef752d028fad2f3217ce0950f4"
            branch = "minimax_h3"
            overlay = "worktrees/InfiniLM-qwen-video"
        }
    }
    infiniCore_overlay_files = @(
        "include/infinicore/ops.hpp",
        "include/infinicore/ops/gated_residual.hpp",
        "python/infinicore/__init__.py",
        "python/infinicore/ops/gated_residual.py",
        "src/infinicore/ops/gated_residual/gated_residual.cc",
        "src/infinicore/ops/gated_residual/gated_residual_hygon.cc",
        "src/infinicore/ops/multi_head_attention/mha.cc",
        "src/infinicore/ops/multi_head_attention/mha_hygon_infiniop.cc",
        "src/infinicore/pybind11/ops.hpp",
        "src/infinicore/pybind11/ops/gated_residual.hpp"
    )
    infiniLM_overlay_files = @(
        "csrc/models/minimax_h3/minimax_h3_transformer.cpp",
        "csrc/models/qwen3_5/qwen3_5_vision.cpp",
        "csrc/models/qwen3_5/qwen3_5_vision.hpp"
    )
}

$manifestPath = Join-Path $stageRoot "MIGRATION_MANIFEST.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

$hashRows = Get-ChildItem -LiteralPath $stageRoot -File -Recurse -Force |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [IO.Path]::GetRelativePath($stageRoot, $_.FullName).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }

$hashRows | Set-Content -LiteralPath (Join-Path $stageRoot "SOURCE_SHA256SUMS") -Encoding ascii

Push-Location $stageRoot
try {
    & tar -czf $archivePath Projects scripts MIGRATION_MANIFEST.json SOURCE_SHA256SUMS
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$archiveHash  $(Split-Path -Leaf $archivePath)" |
    Set-Content -LiteralPath $checksumPath -Encoding ascii

$archive = Get-Item -LiteralPath $archivePath
[PSCustomObject]@{
    Archive = $archive.FullName
    SizeMiB = [math]::Round($archive.Length / 1MB, 2)
    SHA256 = $archiveHash
    ChecksumFile = $checksumPath
} | Format-List
