@echo off
REM test.bat — Run RAG accuracy test via Docker
REM Note: Requires Docker Compose stack to be running (start.bat)

if not exist ".env" (
    echo Error: .env file not found
    exit /b 1
)

echo.
echo Starting RAG Accuracy Test...
echo Note: This requires the Docker stack to be running (start.bat)
echo.

docker compose -f docker/docker-compose.yml --profile test run --rm rag_test

if errorlevel 1 (
    echo ERROR: Test failed. Make sure Docker stack is running (start.bat)
    pause
    exit /b 1
)

echo.
echo Test completed successfully!
echo Results saved to test_results.json
echo.
pause
