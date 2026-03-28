FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN uv sync --no-install-project

# Copy the rest of the application
COPY . .

# Install the project itself
RUN uv sync

ENV PORT=8000
EXPOSE 8000

# GPU libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# vLLM optimizations for Blackwell (RTX 5080)
ENV VLLM_USE_V1=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV VLLM_FLASH_ATTN_VERSION=2
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
