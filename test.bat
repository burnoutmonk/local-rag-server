@echo off
REM test.bat — Run RAG accuracy test via Docker
REM Note: Requires Docker Compose stack to be running (start.bat)

setlocal enabledelayedexpansion

if not exist ".env" (
    echo Error: .env file not found
    exit /b 1
)

:: Load .env to detect GPU mode
for /f "tokens=1,2 delims==" %%a in ('type .env ^| findstr /v "^#"') do (
    set %%a=%%b
)

echo.
echo Starting RAG Accuracy Test...
echo Note: This requires the Docker stack to be running (start.bat)
echo.

:: Step 1: Start test Qdrant + ingest test documents (isolated from production)
echo [1/2] Starting qdrant_test and ingesting test documents...
if "!CUDA_AVAILABLE!"=="true" (
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test up -d --wait qdrant_test
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test run --no-deps --rm ingest_test
) else (
    docker compose -f docker/docker-compose.yml --profile test up -d --wait qdrant_test
    docker compose -f docker/docker-compose.yml --profile test run --no-deps --rm ingest_test
)

if errorlevel 1 (
    echo.
    echo ERROR: Test ingest failed.
    pause
    exit /b 1
)

:: Step 2: Run the accuracy test against the already-running production API + test Qdrant
echo.
echo [2/2] Running accuracy test...
:: --no-deps: production stack (api, llm) is already running via start.bat
if "!CUDA_AVAILABLE!"=="true" (
    echo GPU mode — using CUDA image for test container
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test run --no-deps --rm rag_test
) else (
    echo CPU mode — using CPU image for test container
    docker compose -f docker/docker-compose.yml --profile test run --no-deps --rm rag_test
)

if errorlevel 1 (
    echo.
    echo ERROR: Test failed. Make sure Docker stack is running (start.bat)
    pause
    exit /b 1
)

echo.
echo Test completed successfully!
echo Results saved to test_results.json
echo.
pause
