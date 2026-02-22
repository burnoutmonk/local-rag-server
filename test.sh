#!/bin/bash
# test.sh — Run RAG accuracy test via Docker
# Note: Requires Docker Compose stack to be running (./run.sh --docker)

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    exit 1
fi

echo ""
echo "Starting RAG Accuracy Test..."
echo "Note: This requires the Docker stack to be running (./run.sh --docker)"
echo ""

docker compose -f docker/docker-compose.yml --profile test run --rm rag_test

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Test failed. Make sure Docker stack is running:"
    echo "  ./run.sh --docker"
    exit 1
fi

echo ""
echo "Test completed successfully!"
echo "Results saved to test_results.json"
echo ""
