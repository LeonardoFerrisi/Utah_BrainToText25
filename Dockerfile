# Use Python 3.12 base image (matching your devcontainer)
FROM python:3.12-bullseye

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up Git LFS
RUN git lfs install

# Copy the project files
COPY . /workspace/

# Make setup script executable
RUN chmod +x /workspace/phase1/setup.sh

# Set working directory to phase1
WORKDIR /workspace/phase1

# Create virtual environment and install dependencies
RUN python -m venv .b2txt2025_docker && \
    . .b2txt2025_docker/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Create a non-root user
RUN useradd -m -s /bin/bash vscode && \
    chown -R vscode:vscode /workspace

USER vscode

# Set environment variables
ENV PATH="/workspace/phase1/.b2txt2025_docker/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/phase1/.b2txt2025_docker"

# Default command
CMD ["/bin/bash"]