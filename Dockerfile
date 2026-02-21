FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp (with CUDA if available)
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp --depth=1

ARG CUDA_AVAILABLE=false
RUN if [ "$CUDA_AVAILABLE" = "true" ]; then         echo "Building with CUDA support..." &&         cmake -B /llama.cpp/build -S /llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON;     else         echo "Building CPU only..." &&         cmake -B /llama.cpp/build -S /llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF;     fi     && cmake --build /llama.cpp/build --config Release -j$(nproc)

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy app files
COPY config.py ingest.py rag_api.py download_model.py benchmark.py ./
COPY templates/ templates/

EXPOSE 8000 8080

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]