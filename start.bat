@echo off

:: Load .env file
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
)

:: Choose compose files based on CUDA_AVAILABLE
if "%CUDA_AVAILABLE%"=="true" (
    echo Starting with GPU support...
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
) else (
    echo Starting in CPU mode...
    docker compose up -d
)

echo Waiting for services to be ready...
:wait
docker inspect rag_ready --format "{{.State.Status}}" 2>nul | findstr "exited" >nul
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait
)

echo Opening browser...
start http://localhost:8000