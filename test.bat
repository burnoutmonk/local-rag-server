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

:: Use same compose files as start.bat so rag_test uses the correct image
if "!CUDA_AVAILABLE!"=="true" (
    echo GPU mode — using CUDA image for test container
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test run --rm rag_test
) else (
    echo CPU mode — using CPU image for test container
    docker compose -f docker/docker-compose.yml --profile test run --rm rag_test
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
