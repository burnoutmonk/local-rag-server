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

# Step 1: Start test Qdrant + ingest test documents (isolated from production)
echo "[1/2] Starting qdrant_test and ingesting test documents..."
if [ "$GPU_MODE" = "true" ]; then
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test up -d --wait qdrant_test
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test run --no-deps --rm ingest_test
else
    docker compose -f docker/docker-compose.yml --profile test up -d --wait qdrant_test
    docker compose -f docker/docker-compose.yml --profile test run --no-deps --rm ingest_test
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Test ingest failed."
    exit 1
fi

# Step 2: Run accuracy test against the already-running production API + test Qdrant
echo ""
echo "[2/2] Running accuracy test..."
if [ "$GPU_MODE" = "true" ]; then
    echo "GPU mode — using CUDA image for test container"
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test run --no-deps --rm rag_test
else
    echo "CPU mode — using CPU image for test container"
    docker compose -f docker/docker-compose.yml --profile test run --no-deps --rm rag_test
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
