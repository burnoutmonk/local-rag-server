#!/bin/bash
# test.sh — Run RAG accuracy test via Docker
# Note: Requires Docker Compose stack to be running (./run.sh --docker)

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Load .env to detect GPU mode (same logic as run.sh)
if grep -q "CUDA_AVAILABLE=true" .env 2>/dev/null; then
    GPU_MODE=true
else
    GPU_MODE=false
fi

echo ""
echo "Starting RAG Accuracy Test..."
echo "Note: This requires the Docker stack to be running (./run.sh --docker)"
echo ""

if [ "$GPU_MODE" = "true" ]; then
    echo "GPU mode — using CUDA image for test container"
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test run --rm rag_test
else
    echo "CPU mode — using CPU image for test container"
    docker compose -f docker/docker-compose.yml --profile test run --rm rag_test
fi

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
