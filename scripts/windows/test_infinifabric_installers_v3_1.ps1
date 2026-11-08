$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$artifactDir = Join-Path $repoRoot 'artifacts\infinifabric-migration-v3'

$installers = @(
    Join-Path $artifactDir 'install_infinicore_v3_2.sh'
    Join-Path $artifactDir 'install_infinilm_v3_2.sh'
)

foreach ($installer in $installers) {
    if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
        throw "Missing AFS-safe installer: $installer"
    }

    $content = Get-Content -LiteralPath $installer -Raw

    if ($content -notmatch 'INSTALLER_VERSION="3\.2-afs-no-dev-fd"') {
        throw "Installer version marker is missing: $installer"
    }
    if ($content -match 'find "\$\{TARGET_DIR\}" -mindepth') {
        throw "Installer still scans the existing AFS tree before backup: $installer"
    }
    if ($content -match '/dev/fd' -or $content -match 'exec > >\(') {
        throw "Installer still depends on unavailable /dev/fd process substitution: $installer"
    }
    if ($content -notmatch 'log_line\(\)' -or
        $content -notmatch 'tee -a "\$\{LOG_FILE\}"') {
        throw "Installer does not provide /dev/fd-free live stage logging: $installer"
    }
    if ($content -notmatch 'trap cleanup EXIT') {
        throw "Installer does not retain failure recovery in the main shell: $installer"
    }
    if ($content -notmatch 'Extracting into staging directory') {
        throw "Installer does not report staging extraction: $installer"
    }
    if ($content -notmatch 'Moving existing target to recoverable backup') {
        throw "Installer does not report the potentially slow backup operation: $installer"
    }
    if ($content -notmatch 'source_checksum=OK' -or
        $content -notmatch 'binary_artifact_check=OK' -or
        $content -notmatch 'migration: PASSED') {
        throw "Installer completion evidence is incomplete: $installer"
    }
}

Write-Output 'AFS-safe installer structure: PASSED'

$resumeInstaller = Join-Path $artifactDir 'resume_infinicore_v3_2.sh'
if (-not (Test-Path -LiteralPath $resumeInstaller -PathType Leaf)) {
    throw "Missing resumable promotion script: $resumeInstaller"
}
$resumeContent = Get-Content -LiteralPath $resumeInstaller -Raw
if ($resumeContent -match '/dev/fd|exec > >\(') {
    throw "Resume script depends on unavailable /dev/fd"
}
if ($resumeContent -notmatch 'staged_source_checksum=OK' -or
    $resumeContent -notmatch 'InfiniCore migration: PASSED') {
    throw "Resume script does not require prior verification evidence or emit completion evidence"
}
if ($resumeContent -notmatch 'InfiniCore\.partial') {
    throw "Resume script does not preserve the recoverable backup path"
}
Write-Output 'Resumable promotion structure: PASSED'

$recoveryInstaller = Join-Path $artifactDir 'recover_infinicore_v3_3.sh'
if (-not (Test-Path -LiteralPath $recoveryInstaller -PathType Leaf)) {
    throw "Missing incomplete-target recovery script: $recoveryInstaller"
}
$recoveryContent = Get-Content -LiteralPath $recoveryInstaller -Raw
if ($recoveryContent -match '/dev/fd|exec > >\(' -or $recoveryContent -match '\bfind\b') {
    throw "Recovery script uses an unsupported or slow filesystem mechanism"
}
if ($recoveryContent -notmatch 'InfiniCore\.incomplete' -or
    $recoveryContent -notmatch 'InfiniCore/xmake\.lua' -or
    $recoveryContent -notmatch 'recovery_action=promoted_verified_staging') {
    throw "Recovery script does not safely preserve an incomplete target and promote a verified staging tree"
}
Write-Output 'Incomplete-target recovery structure: PASSED'
