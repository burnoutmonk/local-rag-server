@echo off
REM test.bat — Run RAG accuracy test via Docker

if not exist ".env" (
    echo Error: .env file not found
    exit /b 1
)

echo.
echo Starting RAG Accuracy Test...
echo.

docker compose -f docker/docker-compose.yml --profile test run --rm rag_test

echo.
echo Test completed. Results saved to test_results.json
echo.
