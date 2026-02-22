#!/bin/bash
# test.sh — Run RAG accuracy test via Docker

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    exit 1
fi

echo ""
echo "Starting RAG Accuracy Test..."
echo ""

docker compose -f docker/docker-compose.yml --profile test run --rm rag_test

echo ""
echo "Test completed. Results saved to test_results.json"
echo ""
