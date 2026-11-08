[CmdletBinding()]
param(
    [string]$Destination = 'E:\projects\minimax-h3-match\相关代码文件\wheel',
    [string]$PythonCommand = 'python'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$TritonFilename = 'triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
$TritonSha256 = 'e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581'
$EinopsFilename = 'einops-0.8.2-py3-none-any.whl'
$PytestFilename = 'pytest-9.0.3-py3-none-any.whl'
$IniconfigFilename = 'iniconfig-2.0.0-py3-none-any.whl'
$PackagingFilename = 'packaging-24.2-py3-none-any.whl'
$PluggyFilename = 'pluggy-1.5.0-py3-none-any.whl'
$PygmentsFilename = 'pygments-2.21.0-py3-none-any.whl'
$FlashRequirement = 'flash-attn==2.6.1+das.opt1.dtk2604.torch290'
$VendorIndex = 'https://pypi.sourcefind.cn/release/dtk/'
$VisionFilename = 'vision-0.24.0.tar.gz'
$VisionUrl = 'https://github.com/pytorch/vision/archive/refs/tags/v0.24.0.tar.gz'

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-PypiWheel(
    [string]$Project,
    [string]$Version,
    [string]$Filename,
    [string]$DestinationDirectory
) {
    $destinationPath = Join-Path $DestinationDirectory $Filename
    $metadataUrl = "https://pypi.org/pypi/$Project/$Version/json"
    $metadata = Invoke-RestMethod -Method Get -Uri $metadataUrl
    $candidate = @($metadata.urls | Where-Object { $_.filename -eq $Filename })
    if ($candidate.Count -ne 1) {
        throw "PyPI metadata did not contain exactly one $Filename candidate"
    }
    $expected = ([string]$candidate[0].digests.sha256).ToLowerInvariant()
    if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
        $actual = Get-Sha256 $destinationPath
        if ($actual -ne $expected) {
            throw "Existing wheel checksum mismatch: $destinationPath expected=$expected actual=$actual"
        }
        Write-Host "OK existing: $Filename"
        return
    }
    $partial = "$destinationPath.partial.$PID"
    try {
        Invoke-WebRequest -Uri $candidate[0].url -OutFile $partial
        $actual = Get-Sha256 $partial
        if ($actual -ne $expected) {
            throw "Downloaded wheel checksum mismatch: $Filename expected=$expected actual=$actual"
        }
        Move-Item -LiteralPath $partial -Destination $destinationPath
        Write-Host "DOWNLOADED: $Filename"
    }
    finally {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
    }
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null

$tritonPath = Join-Path $Destination $TritonFilename
if (-not (Test-Path -LiteralPath $tritonPath -PathType Leaf)) {
    throw "The verified vendor Triton wheel is missing: $tritonPath"
}
$actualTriton = Get-Sha256 $tritonPath
if ($actualTriton -ne $TritonSha256) {
    throw "Triton checksum mismatch: expected=$TritonSha256 actual=$actualTriton"
}
Write-Host "OK existing: $TritonFilename"

Get-PypiWheel -Project 'einops' -Version '0.8.2' -Filename $EinopsFilename -DestinationDirectory $Destination
Get-PypiWheel -Project 'pytest' -Version '9.0.3' -Filename $PytestFilename -DestinationDirectory $Destination
Get-PypiWheel -Project 'iniconfig' -Version '2.0.0' -Filename $IniconfigFilename -DestinationDirectory $Destination
Get-PypiWheel -Project 'packaging' -Version '24.2' -Filename $PackagingFilename -DestinationDirectory $Destination
Get-PypiWheel -Project 'pluggy' -Version '1.5.0' -Filename $PluggyFilename -DestinationDirectory $Destination
Get-PypiWheel -Project 'pygments' -Version '2.21.0' -Filename $PygmentsFilename -DestinationDirectory $Destination

$flashExisting = @(Get-ChildItem -LiteralPath $Destination -File | Where-Object {
    $_.Name -like 'flash_attn-2.6.1+das.opt1.dtk2604.torch290-*.whl'
})
if ($flashExisting.Count -gt 1) {
    throw "Multiple matching FlashAttention wheels exist in $Destination"
}
if ($flashExisting.Count -eq 0) {
    $downloadDirectory = Join-Path $Destination ".flash-attn-download-$PID"
    New-Item -ItemType Directory -Path $downloadDirectory | Out-Null
    try {
        Write-Host "RUN: pip download $FlashRequirement"
        & $PythonCommand -m pip download `
            --dest $downloadDirectory `
            --no-deps `
            --only-binary=:all: `
            --platform manylinux_2_28_x86_64 `
            --implementation cp `
            --python-version 311 `
            --abi cp311 `
            --index-url $VendorIndex `
            $FlashRequirement
        if ($LASTEXITCODE -ne 0) {
            throw "pip download failed for $FlashRequirement with exit code $LASTEXITCODE"
        }
        $downloaded = @(Get-ChildItem -LiteralPath $downloadDirectory -File -Filter '*.whl')
        if ($downloaded.Count -ne 1) {
            throw "Expected one downloaded FlashAttention wheel, found $($downloaded.Count)"
        }
        Move-Item -LiteralPath $downloaded[0].FullName -Destination (Join-Path $Destination $downloaded[0].Name)
        Write-Host "DOWNLOADED: $($downloaded[0].Name)"
    }
    finally {
        if (Test-Path -LiteralPath $downloadDirectory) {
            Get-ChildItem -LiteralPath $downloadDirectory -File -ErrorAction SilentlyContinue |
                Remove-Item -Force
            Remove-Item -LiteralPath $downloadDirectory -Force
        }
    }
}
else {
    Write-Host "OK existing: $($flashExisting[0].Name)"
}

$visionPath = Join-Path $Destination $VisionFilename
if (-not (Test-Path -LiteralPath $visionPath -PathType Leaf)) {
    $partialVision = "$visionPath.partial.$PID"
    try {
        Invoke-WebRequest -Uri $VisionUrl -OutFile $partialVision
        if ((Get-Item -LiteralPath $partialVision).Length -le 0) {
            throw "Downloaded torchvision source archive is empty"
        }
        Move-Item -LiteralPath $partialVision -Destination $visionPath
        Write-Host "DOWNLOADED: $VisionFilename"
    }
    finally {
        if (Test-Path -LiteralPath $partialVision) {
            Remove-Item -LiteralPath $partialVision -Force
        }
    }
}
else {
    Write-Host "OK existing: $VisionFilename"
}

$inputs = @(
    (Join-Path $Destination $EinopsFilename),
    (Join-Path $Destination $PytestFilename),
    (Join-Path $Destination $IniconfigFilename),
    (Join-Path $Destination $PackagingFilename),
    (Join-Path $Destination $PluggyFilename),
    (Join-Path $Destination $PygmentsFilename),
    $tritonPath,
    (Get-ChildItem -LiteralPath $Destination -File | Where-Object {
        $_.Name -like 'flash_attn-2.6.1+das.opt1.dtk2604.torch290-*.whl'
    } | Select-Object -ExpandProperty FullName),
    $visionPath
)
if ($inputs.Count -ne 9) {
    throw "Expected nine runtime inputs, found $($inputs.Count)"
}
$sumLines = foreach ($path in $inputs) {
    "$(Get-Sha256 $path)  $(Split-Path $path -Leaf)"
}
$sumPath = Join-Path $Destination 'SHA256SUMS.hygon-runtime-inputs'
[System.IO.File]::WriteAllLines($sumPath, $sumLines, [System.Text.UTF8Encoding]::new($false))

Write-Host "RESULT: PASSED; input_count=9; destination=$Destination; checksums=$sumPath"
