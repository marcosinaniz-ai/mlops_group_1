# --------------------------------------------------------
# Serving container only
# Packages the FastAPI inference service, NOT training code.
# --------------------------------------------------------

FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# This ensures that logs are flushed immediately, which is crucial
# for real-time monitoring and debugging in production environments.
ENV PYTHONUNBUFFERED=1

# This minimizes unnecessary file writes and avoids stale .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python can resolve absolute imports like `src.api`.
ENV PYTHONPATH=/app

# Copy the lockfile first to leverage layer caching.
# This file is generated with `conda-lock -p linux-64 -f environment.yml`.
COPY conda-lock.yml .

# Install dependencies from lock file and lightweight tooling.
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops conda-lock.yml && \
    apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

# Activate environment binaries in PATH
ENV PATH=/opt/conda/envs/mlops/bin:$PATH

# Copy application code
COPY . .

# Expose default API port
EXPOSE 8000

# For production orchestration health checking
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Start FastAPI with uvicorn
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
