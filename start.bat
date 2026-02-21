@echo off
setlocal enabledelayedexpansion

:: Load .env file
for /f "tokens=1,2 delims==" %%a in ('type .env ^| findstr /v "^#"') do (
    set %%a=%%b
)

echo CUDA_AVAILABLE=!CUDA_AVAILABLE!

:: Check if CUDA mode changed since last run
set LAST_CUDA=
if exist .last_cuda (
    set /p LAST_CUDA=<.last_cuda
)

if "!CUDA_AVAILABLE!"=="!LAST_CUDA!" (
    echo GPU mode unchanged — skipping rebuild.
    set BUILD_FLAG=
) else (
    echo GPU mode changed — rebuilding with --no-cache...
    set BUILD_FLAG=--no-cache
)

:: Save current mode
echo !CUDA_AVAILABLE!>.last_cuda

:: Choose compose files based on CUDA_AVAILABLE
if "!CUDA_AVAILABLE!"=="true" (
    echo Starting with GPU support...
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml build !BUILD_FLAG!
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d
) else (
    echo Starting in CPU mode...
    docker compose -f docker/docker-compose.yml build !BUILD_FLAG!
    docker compose -f docker/docker-compose.yml up -d
)

if errorlevel 1 (
    echo ERROR: Docker Compose failed.
    pause
    exit /b 1
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
pause