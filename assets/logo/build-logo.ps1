$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$taskRoot = $PSScriptRoot
$taskCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Format-Coordinate([double]$Value) {
    return $Value.ToString('0.###', $taskCulture)
}

# Hand-drawn geometry: a 39-unit annulus, interrupted by two 8-unit
# journal gaps. The last band continues into a 32-unit rounded q tail.
$taskMark = @'
    <path d="M38.928 88 A82 82 0 0 1 193.072 88 H148.634 A43 43 0 0 0 83.366 88 Z"/>
    <path d="M36.476 96 H77.934 A43 43 0 0 0 77.934 136 H36.476 A82 82 0 0 1 36.476 96 Z M195.524 96 H154.066 A43 43 0 0 1 154.066 136 H195.524 A82 82 0 0 0 195.524 96 Z"/>
    <path fill="ACCENT" d="M38.928 144 H83.366 A43 43 0 0 0 148.634 144 H193.072 A82 82 0 0 1 184.183 161.555 L216.314 193.686 A16 16 0 0 1 193.686 216.314 L161.555 184.183 A82 82 0 0 1 38.928 144 Z"/>
'@

# Convert locally installed Segoe UI Semibold to ordinary SVG paths.
# The resulting assets contain no text elements or font dependency.
$taskFont = [System.Drawing.FontFamily]::new('Segoe UI Semibold')
$taskOutline = [System.Drawing.Drawing2D.GraphicsPath]::new()
try {
    $taskOutline.AddString('qagents', $taskFont, [int][System.Drawing.FontStyle]::Regular,
        [single]144, [System.Drawing.PointF]::new(0, 0),
        [System.Drawing.StringFormat]::GenericTypographic)
    $taskBounds = $taskOutline.GetBounds()
    $taskWidth = [int][Math]::Ceiling(282 + $taskBounds.Width + 34)
    $taskMatrix = [System.Drawing.Drawing2D.Matrix]::new()
    try {
        $taskMatrix.Translate([single](282 - $taskBounds.X),
            [single](128 - $taskBounds.Height / 2 - $taskBounds.Y))
        $taskOutline.Transform($taskMatrix)
    } finally {
        $taskMatrix.Dispose()
    }

    $taskPoints = $taskOutline.PathPoints
    $taskTypes = $taskOutline.PathTypes
    $taskPath = [System.Text.StringBuilder]::new()
    $taskIndex = 0
    while ($taskIndex -lt $taskPoints.Length) {
        $taskType = $taskTypes[$taskIndex] -band 7
        if ($taskType -eq 0 -or $taskType -eq 1) {
            $taskCommand = if ($taskType -eq 0) { 'M' } else { 'L' }
            [void]$taskPath.Append($taskCommand + (Format-Coordinate $taskPoints[$taskIndex].X) +
                ' ' + (Format-Coordinate $taskPoints[$taskIndex].Y) + ' ')
            $taskLast = $taskIndex
            $taskIndex++
        } elseif ($taskType -eq 3) {
            [void]$taskPath.Append('C')
            for ($taskControl = 0; $taskControl -lt 3; $taskControl++) {
                $taskPoint = $taskPoints[$taskIndex + $taskControl]
                [void]$taskPath.Append((Format-Coordinate $taskPoint.X) + ' ' +
                    (Format-Coordinate $taskPoint.Y) + ' ')
            }
            $taskLast = $taskIndex + 2
            $taskIndex += 3
        } else {
            throw "Unsupported outline segment: $taskType"
        }
        if (($taskTypes[$taskLast] -band 128) -ne 0) {
            [void]$taskPath.Append('Z ')
        }
    }
    $taskWordmark = $taskPath.ToString().Trim()
} finally {
    $taskOutline.Dispose()
    $taskFont.Dispose()
}

function Write-LogoSvg([string]$Name, [string]$Primary, [string]$Accent,
    [string]$Neutral, [bool]$Lockup) {
    $taskSvgWidth = if ($Lockup) { $taskWidth } else { 256 }
    $taskDescription = if ($Lockup) {
        'A q formed from three journal bands, followed by the lowercase qagents wordmark.'
    } else {
        'A q formed from three journal bands; the final teal band extends into a rounded tail.'
    }
    $taskSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" width="$taskSvgWidth" height="256" viewBox="0 0 $taskSvgWidth 256" role="img" aria-labelledby="logo-title logo-desc">
  <title id="logo-title">qagents</title>
  <desc id="logo-desc">$taskDescription</desc>
  <g id="mark" fill="$Primary">
$($taskMark.Replace('ACCENT', $Accent))
  </g>
"@
    if ($Lockup) {
        $taskSvg += "`n  <path id=`"wordmark`" fill=`"$Neutral`" fill-rule=`"evenodd`" d=`"$taskWordmark`"/>"
    }
    $taskSvg += "`n</svg>`n"
    [System.IO.File]::WriteAllText((Join-Path $taskRoot $Name), $taskSvg,
        [System.Text.UTF8Encoding]::new($false))
}

Write-LogoSvg 'qagents-icon.svg' '#3B3F9E' '#008D82' '' $false
Write-LogoSvg 'qagents-logo.svg' '#3B3F9E' '#008D82' '#18243B' $true
Write-LogoSvg 'qagents-logo-dark.svg' '#A5AAFF' '#45D6BF' '#F0F3FA' $true

if (Get-Command magick -ErrorAction SilentlyContinue) {
    foreach ($taskAsset in @(
        @{ Name = 'qagents-icon'; Background = '#FFFFFF' },
        @{ Name = 'qagents-logo'; Background = '#FFFFFF' },
        @{ Name = 'qagents-logo-dark'; Background = '#0D1117' }
    )) {
        $taskInput = Join-Path $taskRoot ($taskAsset.Name + '.svg')
        $taskOutput = Join-Path $taskRoot ($taskAsset.Name + '.png')
        & magick -background $taskAsset.Background -density 384 $taskInput -resize '512x' -alpha remove -alpha off "PNG24:$taskOutput"
        if ($LASTEXITCODE -ne 0) { throw "Rendering failed: $taskInput" }
    }
} else {
    Write-Warning 'ImageMagick is unavailable; SVGs were saved without PNG previews.'
}

Write-Output "Saved three SVGs. Lockup viewBox: 0 0 $taskWidth 256."
