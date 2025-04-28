@echo off
setlocal

if "%~1"=="" (
  echo Usage: run.bat ^<32-char MD5^> [num_threads]
  exit /b 1
)
set HASH=%~1
set THR=%~2
if "%THR%"=="" set THR=8

set GPU_EXE=..\bin\gpu_crack.exe
set CPU_EXE=..\bin\cpu_crack.exe

echo Running GPU...
%GPU_EXE% %HASH% > gpu.txt

:: Parse "GPU elapsed     : 0.000 s"
for /f "tokens=2 delims=:" %%A in ('findstr /C:"GPU elapsed" gpu.txt') do (
    for /f "tokens=1" %%B in ("%%A") do set GPU_TIME=%%B
)

echo Running CPU with %THR% threads...
%CPU_EXE% %HASH% -t %THR% > cpu.txt

:: Parse "CPU time (8 threads): 0.0008769 s"
for /f "tokens=2 delims=:" %%A in ('findstr /C:"CPU time" cpu.txt') do (
    for /f "tokens=1" %%B in ("%%A") do set CPU_TIME=%%B
)

:: Write CSV
(
  echo program,time,threads
  echo GPU,%GPU_TIME%,
  echo CPU,%CPU_TIME%,%THR%
) > results.csv

echo.
echo === Results ===
type results.csv

endlocal
