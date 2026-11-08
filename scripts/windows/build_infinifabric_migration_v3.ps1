param(
    [string]$ProjectRoot = "E:\projects\Codex_projects\MiniMax-H3"
)

$ErrorActionPreference = "Stop"

$reposRoot = Join-Path $ProjectRoot "repos"
$worktreesRoot = Join-Path $ProjectRoot "worktrees"
$outputRoot = Join-Path $ProjectRoot "artifacts\infinifabric-migration-v3"
$stageRoot = Join-Path $outputRoot "stage"

if (Test-Path -LiteralPath $outputRoot) {
    throw "Output already exists: $outputRoot"
}

New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null

function Copy-CleanSourceTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    & robocopy $Source $Destination /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
        /XD .git .xmake build __pycache__ .pytest_cache .mypy_cache `
        /XF .git "*.pyc" "*.pyo" "*.so" "*.a" "*.o" "*.obj" "*.dll" "*.pyd" "*.lib" "*.exp"

    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code ${LASTEXITCODE}: $Source"
    }
}

function Write-LfText {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Lines
    )

    $text = ($Lines -join "`n") + "`n"
    [IO.File]::WriteAllText($Path, $text, [Text.UTF8Encoding]::new($false))
}

function Write-ProjectChecksums {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectName
    )

    $projectPath = Join-Path $stageRoot $ProjectName
    $lines = Get-ChildItem -LiteralPath $projectPath -File -Recurse -Force |
        Sort-Object FullName |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($stageRoot, $_.FullName).Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relative"
        }

    Write-LfText -Path (Join-Path $stageRoot "${ProjectName}.SHA256SUMS") -Lines $lines
}

Copy-CleanSourceTree `
    -Source (Join-Path $reposRoot "InfiniCore") `
    -Destination (Join-Path $stageRoot "InfiniCore")

Copy-CleanSourceTree `
    -Source (Join-Path $worktreesRoot "InfiniCore-h3-mha") `
    -Destination (Join-Path $stageRoot "InfiniCore")

Copy-CleanSourceTree `
    -Source (Join-Path $reposRoot "InfiniLM") `
    -Destination (Join-Path $stageRoot "InfiniLM")

Copy-CleanSourceTree `
    -Source (Join-Path $worktreesRoot "InfiniLM-qwen-video") `
    -Destination (Join-Path $stageRoot "InfiniLM")

Write-ProjectChecksums -ProjectName "InfiniCore"
Write-ProjectChecksums -ProjectName "InfiniLM"

$manifest = [ordered]@{
    format = 3
    created_at = (Get-Date).ToString("o")
    target = "InfiniFabric 8-card / DTK 26.04"
    packaging = "split clean source archives"
    excluded = @(
        ".git", ".xmake", "build", "__pycache__", ".pytest_cache", ".mypy_cache",
        "*.pyc", "*.pyo", "*.so", "*.a", "*.o", "*.obj", "*.dll", "*.pyd", "*.lib", "*.exp"
    )
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
}

$manifestPath = Join-Path $outputRoot "MIGRATION_MANIFEST.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 8
Write-LfText -Path $manifestPath -Lines @($manifestJson)

$rows = @()
foreach ($projectName in @("InfiniCore", "InfiniLM")) {
    $archiveName = "${projectName}-infinifabric-v3.tar.gz"
    $archivePath = Join-Path $outputRoot $archiveName

    Push-Location $stageRoot
    try {
        & tar -czf $archivePath "${projectName}.SHA256SUMS" $projectName
        if ($LASTEXITCODE -ne 0) {
            throw "tar failed for $projectName with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-LfText -Path "${archivePath}.sha256" -Lines @("$archiveHash  $archiveName")

    $projectFiles = Get-ChildItem -LiteralPath (Join-Path $stageRoot $projectName) -File -Recurse -Force
    $binaryLeaks = @($projectFiles | Where-Object {
        $_.Name -match '\.(so|a|o|obj|dll|pyd|lib|exp)$'
    })
    if ($binaryLeaks.Count -ne 0) {
        throw "Binary artifacts leaked into ${projectName}: $($binaryLeaks.FullName -join ', ')"
    }

    $rows += [PSCustomObject]@{
        Project = $projectName
        Files = $projectFiles.Count
        SourceMiB = [math]::Round((($projectFiles | Measure-Object Length -Sum).Sum) / 1MB, 2)
        ArchiveMiB = [math]::Round((Get-Item -LiteralPath $archivePath).Length / 1MB, 2)
        SHA256 = $archiveHash
    }
}

$rows | Format-Table -AutoSize
