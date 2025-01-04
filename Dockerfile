# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DGLBACKEND=pytorch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    python-is-python3 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install CUDA toolkit
RUN pip install nvidia-cudnn-cu12==8.9.2.26

# Create DGL config directory and file
RUN mkdir -p /root/.dgl && \
    echo '{"backend":"pytorch"}' > /root/.dgl/config.json

# Install PyTorch and DGL with CUDA support first
RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies in stages to better handle failures
RUN pip install -r /tmp/requirements.txt

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Set default command
CMD ["bash"]