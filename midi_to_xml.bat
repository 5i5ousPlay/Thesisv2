@echo off
setlocal enabledelayedexpansion

REM Set the path to your MuseScore executable
set mscore_path="C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

REM Set the folder where your MIDI files are stored
set midi_folder="C:\.School\Thesisv2\Music Database\bach_cello_suites\Suite No. 1 in G major"

REM Set the output folder for MusicXML files
set output_folder="C:\.School\Thesisv2\Music Database\bach_cello_suites\xml"

for %%f in (%midi_folder%\*.mid) do (
    set input_file=%%f
    set output_file=%output_folder%\%%~nf.musicxml
    %mscore_path% -o "!output_file!" "!input_file!"
)

echo Conversion complete!
pause
