FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo ">>> Building llama.cpp: CPU only <<<"
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp --depth=1 \
    && cmake -B /llama.cpp/build -S /llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build /llama.cpp/build --config Release -j$(nproc)
RUN echo ">>> llama.cpp CPU build complete <<<"
RUN test -f /llama.cpp/build/bin/llama-server || (echo "ERROR: llama-server was not built!" && exit 1)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY config.py ingest.py rag_api.py download_model.py benchmark.py ./
COPY templates/ templates/

EXPOSE 8000 8080

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]