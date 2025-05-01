@echo off
setlocal

REM Clean previous output
if exist timings.csv del timings.csv

REM CSV header
echo threads,time_seconds > timings.csv

REM Loop through thread counts 1 to 16
for /L %%T in (1,1,16) do (
    echo Running with %%T threads...

    REM Use echo + pipe to pass password "amijan" into the program
    echo amijan | out.exe 6 %%T >> timings.csv
)

echo Done! Results saved to timings.csv
pause