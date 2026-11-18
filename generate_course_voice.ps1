$ErrorActionPreference = 'Stop'
$courseRoot = $PSScriptRoot
$courseOut = Join-Path $courseRoot 'output\course_delivery'
$courseAudio = Join-Path $courseOut 'audio'
New-Item -ItemType Directory -Force -Path $courseAudio | Out-Null
$courseScenes = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $courseOut 'storyboard.json') | ConvertFrom-Json
Add-Type -AssemblyName System.Speech
$courseSynth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
$courseSynth.SelectVoice('Microsoft Huihui Desktop')
$courseSynth.Rate = 0
$courseSynth.Volume = 100
try {
    foreach ($courseScene in $courseScenes) {
        $courseParagraphIndex = 0
        foreach ($courseParagraph in $courseScene.narration) {
            $courseWave = Join-Path $courseAudio ("voice_{0}_{1}.wav" -f $courseScene.index, $courseParagraphIndex)
            $courseSynth.SetOutputToWaveFile($courseWave)
            $courseSynth.Speak($courseParagraph)
            $courseSynth.SetOutputToNull()
            Write-Output ("Voice {0}:{1} complete" -f $courseScene.index, $courseParagraphIndex)
            $courseParagraphIndex += 1
        }
    }
} finally {
    $courseSynth.Dispose()
}
