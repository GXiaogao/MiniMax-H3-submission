$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$installer = Join-Path $repoRoot 'artifacts\infinifabric-toolchain-v1\install_infinifabric_toolchain_v1.sh'

if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "Installer is missing: $installer"
}

$content = Get-Content -LiteralPath $installer -Raw
$requiredPatterns = @(
    'IFAB_AFS_HOME',
    'infinifabric-toolchain-v1\.tar\.gz',
    'sha256sum --check --quiet',
    'ELF 64-bit',
    'x86-64|x86_64',
    'packages/b/boost/xmake\.lua',
    'packages/p/pybind11/xmake\.lua',
    '--network=private',
    '--pkg_searchdirs=',
    'toolchain migration: PASSED'
)
foreach ($pattern in $requiredPatterns) {
    if ($content -notmatch $pattern) {
        throw "Installer is missing required behavior matching: $pattern"
    }
}

if ($content -match '/dev/fd|exec > >\(' -or $content -match '\bfind\b') {
    throw 'Installer uses an unsupported or slow filesystem mechanism'
}

$bytes = [IO.File]::ReadAllBytes((Resolve-Path $installer))
if (($bytes | Where-Object { $_ -eq 13 }).Count -ne 0) {
    throw 'Installer contains CRLF characters'
}

Write-Output 'Offline toolchain installer structure: PASSED'
