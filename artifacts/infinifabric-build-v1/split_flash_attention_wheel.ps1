[CmdletBinding()]
param(
    [string]$Source = 'E:\projects\minimax-h3-match\相关代码文件\wheel\flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl',
    [string]$Destination = 'E:\projects\minimax-h3-match\相关代码文件\wheel\flash_attn_parts',
    [ValidateRange(1, 120)]
    [int]$ChunkMiB = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Source wheel is missing: $Source"
}
if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists; refusing to overwrite it: $Destination"
}

$sourceItem = Get-Item -LiteralPath $Source
$fullSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLowerInvariant()
$expectedFullSha256 = '8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1'
if ($fullSha256 -ne $expectedFullSha256) {
    throw "FlashAttention wheel checksum mismatch: expected=$expectedFullSha256 actual=$fullSha256"
}

$stage = "$Destination.stage.$PID"
New-Item -ItemType Directory -Path $stage | Out-Null
$chunkBytes = $ChunkMiB * 1MB
$buffer = [byte[]]::new($chunkBytes)
$parts = [System.Collections.Generic.List[object]]::new()
$input = [System.IO.FileStream]::new(
    $sourceItem.FullName,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::Read
)
try {
    $index = 0
    while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
        $partName = "$($sourceItem.Name).part-{0:D3}" -f $index
        $partPath = Join-Path $stage $partName
        $output = [System.IO.FileStream]::new(
            $partPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            $output.Write($buffer, 0, $read)
            $output.Flush()
        }
        finally {
            $output.Dispose()
        }
        $partSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $partPath).Hash.ToLowerInvariant()
        $parts.Add([pscustomobject]@{
            filename = $partName
            size = $read
            part_sha256 = $partSha256
        })
        Write-Host "CREATED: $partName bytes=$read sha256=$partSha256"
        $index++
    }
}
finally {
    $input.Dispose()
}

if ($parts.Count -lt 2) {
    throw "Expected a multi-part result, got $($parts.Count) part(s)"
}
$totalSize = ($parts | Measure-Object -Property size -Sum).Sum
if ($totalSize -ne $sourceItem.Length) {
    throw "Part size sum mismatch: source=$($sourceItem.Length) parts=$totalSize"
}

$manifest = [ordered]@{
    schema_version = 1
    source_filename = $sourceItem.Name
    full_size = $sourceItem.Length
    full_sha256 = $fullSha256
    chunk_mib = $ChunkMiB
    part_count = $parts.Count
    parts = $parts
}
$manifestPath = Join-Path $stage 'flash-wheel-manifest.json'
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM

$sumLines = foreach ($part in $parts) {
    "$($part.part_sha256)  $($part.filename)"
}
$sumLines += "$fullSha256  $($sourceItem.Name)"
$sumPath = Join-Path $stage 'SHA256SUMS.flash-parts'
[System.IO.File]::WriteAllLines($sumPath, $sumLines, [System.Text.UTF8Encoding]::new($false))

Move-Item -LiteralPath $stage -Destination $Destination
Write-Host "RESULT: PASSED; part_count=$($parts.Count); full_sha256=$fullSha256; destination=$Destination"
