[CmdletBinding()]
param(
    [string]$Destination = 'E:\projects\minimax-h3-match\相关代码文件\new_wheels',
    [string]$LocalCache = 'E:\projects\minimax-h3-match\相关代码文件\wheels',
    [switch]$Offline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Wheels = @(
    [pscustomobject]@{ Project = 'annotated-doc'; Version = '0.0.5'; Filename = 'annotated_doc-0.0.5-py3-none-any.whl'; Sha256 = '117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101' }
    [pscustomobject]@{ Project = 'click'; Version = '8.4.2'; Filename = 'click-8.4.2-py3-none-any.whl'; Sha256 = 'e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76' }
    [pscustomobject]@{ Project = 'h11'; Version = '0.16.0'; Filename = 'h11-0.16.0-py3-none-any.whl'; Sha256 = '63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86' }
    [pscustomobject]@{ Project = 'hf-xet'; Version = '1.5.2'; Filename = 'hf_xet-1.5.2-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'; Sha256 = 'db78c39c83d6279daddc98e2238f373ab8980685556d42472b4ec51abcf03e8c' }
    [pscustomobject]@{ Project = 'httpcore'; Version = '1.0.9'; Filename = 'httpcore-1.0.9-py3-none-any.whl'; Sha256 = '2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55' }
    [pscustomobject]@{ Project = 'huggingface-hub'; Version = '1.32.0'; Filename = 'huggingface_hub-1.32.0-py3-none-any.whl'; Sha256 = 'b0c7c80561969d9cdacdd55fce67ba9584cca0b9d4ea80957a3a5c1445fac5c8' }
    [pscustomobject]@{ Project = 'imageio'; Version = '2.34.2'; Filename = 'imageio-2.34.2-py3-none-any.whl'; Sha256 = 'a0bb27ec9d5bab36a9f4835e51b21d2cb099e1f78451441f94687ff3404b79f8' }
    [pscustomobject]@{ Project = 'markdown-it-py'; Version = '4.2.0'; Filename = 'markdown_it_py-4.2.0-py3-none-any.whl'; Sha256 = '9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a' }
    [pscustomobject]@{ Project = 'mdurl'; Version = '0.1.2'; Filename = 'mdurl-0.1.2-py3-none-any.whl'; Sha256 = '84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8' }
    [pscustomobject]@{ Project = 'pygments'; Version = '2.21.0'; Filename = 'pygments-2.21.0-py3-none-any.whl'; Sha256 = '2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9' }
    [pscustomobject]@{ Project = 'regex'; Version = '2026.9.29'; Filename = 'regex-2026.9.29-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl'; Sha256 = '612b709381c0355b70d89cdb51b7f670591ed5cbbc0e3b5337488019dc667b65' }
    [pscustomobject]@{ Project = 'rich'; Version = '15.0.0'; Filename = 'rich-15.0.0-py3-none-any.whl'; Sha256 = '33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb' }
    [pscustomobject]@{ Project = 'safetensors'; Version = '0.8.0'; Filename = 'safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'; Sha256 = 'fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774' }
    [pscustomobject]@{ Project = 'shellingham'; Version = '1.5.4'; Filename = 'shellingham-1.5.4-py2.py3-none-any.whl'; Sha256 = '7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686' }
    [pscustomobject]@{ Project = 'tifffile'; Version = '2024.8.30'; Filename = 'tifffile-2024.8.30-py3-none-any.whl'; Sha256 = '8bc59a8f02a2665cd50a910ec64961c5373bee0b8850ec89d3b7b485bf7be7ad' }
    [pscustomobject]@{ Project = 'tokenizers'; Version = '0.23.3'; Filename = 'tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'; Sha256 = '376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87' }
    [pscustomobject]@{ Project = 'transformers'; Version = '5.20.2'; Filename = 'transformers-5.20.2-py3-none-any.whl'; Sha256 = '7ae75dbc1fb3922aff1b6c97b9e5479d6cf26fd86a8866bfb71324190479f807' }
    [pscustomobject]@{ Project = 'typer'; Version = '0.27.3'; Filename = 'typer-0.27.3-py3-none-any.whl'; Sha256 = 'e50022f28b82a86313e54501317a1db64bf8f8d036ff8cfe5ca7e47675454aff' }
)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null

foreach ($wheel in $Wheels) {
    $destinationPath = Join-Path $Destination $wheel.Filename
    if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
        $actual = Get-Sha256 $destinationPath
        if ($actual -ne $wheel.Sha256) {
            throw "Existing destination file has the wrong SHA256: $destinationPath expected=$($wheel.Sha256) actual=$actual"
        }
        Write-Host "OK existing: $($wheel.Filename)"
        continue
    }

    $cachePath = Join-Path $LocalCache $wheel.Filename
    if (Test-Path -LiteralPath $cachePath -PathType Leaf) {
        $cacheHash = Get-Sha256 $cachePath
        if ($cacheHash -ne $wheel.Sha256) {
            throw "Local cache file has the wrong SHA256: $cachePath expected=$($wheel.Sha256) actual=$cacheHash"
        }
        Copy-Item -LiteralPath $cachePath -Destination $destinationPath
        Write-Host "COPIED from local cache: $($wheel.Filename)"
        continue
    }

    if ($Offline) {
        throw "Required wheel is absent from the local cache while -Offline is set: $($wheel.Filename)"
    }

    $metadataUrl = "https://pypi.org/pypi/$($wheel.Project)/$($wheel.Version)/json"
    Write-Host "QUERY $metadataUrl"
    $metadata = Invoke-RestMethod -Method Get -Uri $metadataUrl
    $candidates = @($metadata.urls | Where-Object { $_.filename -eq $wheel.Filename })
    if ($candidates.Count -ne 1) {
        throw "PyPI metadata did not contain exactly one matching file: $($wheel.Filename) count=$($candidates.Count)"
    }
    $candidate = $candidates[0]
    $metadataHash = ([string]$candidate.digests.sha256).ToLowerInvariant()
    if ($metadataHash -ne $wheel.Sha256) {
        throw "PyPI metadata SHA256 mismatch for $($wheel.Filename): expected=$($wheel.Sha256) metadata=$metadataHash"
    }

    $partialPath = "$destinationPath.partial.$PID"
    try {
        Invoke-WebRequest -Uri $candidate.url -OutFile $partialPath
        $downloadHash = Get-Sha256 $partialPath
        if ($downloadHash -ne $wheel.Sha256) {
            throw "Downloaded file SHA256 mismatch for $($wheel.Filename): expected=$($wheel.Sha256) actual=$downloadHash"
        }
        Move-Item -LiteralPath $partialPath -Destination $destinationPath
        Write-Host "DOWNLOADED: $($wheel.Filename)"
    }
    finally {
        if (Test-Path -LiteralPath $partialPath) {
            Remove-Item -LiteralPath $partialPath -Force
        }
    }
}

$sumLines = foreach ($wheel in $Wheels) {
    "$($wheel.Sha256)  $($wheel.Filename)"
}
$sumPath = Join-Path $Destination 'SHA256SUMS'
[System.IO.File]::WriteAllLines($sumPath, $sumLines, [System.Text.UTF8Encoding]::new($false))

Write-Host "RESULT: PASSED; wheel_count=$($Wheels.Count); destination=$Destination"
