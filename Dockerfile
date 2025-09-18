# Industrial Quality Control Computer Vision System
# Multi-stage Docker build for production deployment

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgflags-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libhdf5-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Stage 2: Python dependencies
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorRT (for edge optimization)
RUN pip install nvidia-tensorrt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs temp data/images backups

# Download pre-trained models (if available)
RUN python scripts/download_models.py || echo "Model download failed - will use default models"

# Set proper permissions
RUN chmod +x scripts/*.py

# Create non-root user for security
RUN useradd -m -u 1000 qcuser && \
    chown -R qcuser:qcuser /app

# Switch to non-root user
USER qcuser

# Stage 4: Production
FROM application as production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose ports
EXPOSE 5000 9090

# Set environment variables for production
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "--timeout", "120", "app:app"]

# Stage 5: Development (optional)
FROM application as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    ipykernel \
    pytest-xdist \
    pytest-benchmark \
    memory-profiler

# Install debugging tools
RUN apt-get update && apt-get install -y \
    htop \
    vim \
    nano \
    git \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Switch back to qcuser
USER qcuser

# Expose additional ports for development
EXPOSE 8888

# Development command
CMD ["python", "app.py"]

# Multi-architecture build support
# To build for different architectures:
# docker buildx build --platform linux/amd64,linux/arm64 -t qc-system:latest .

# Build arguments for customization
ARG MODEL_VERSION=latest
ARG TENSORRT_VERSION=8.6.0
ARG CUDA_VERSION=11.8

# Labels for metadata
LABEL maintainer="your-email@company.com"
LABEL version="1.0.0"
LABEL description="Industrial Quality Control Computer Vision System"
LABEL vendor="Your Company"
LABEL org.opencontainers.image.source="https://github.com/yourusername/industrial-qc-cv-system"
