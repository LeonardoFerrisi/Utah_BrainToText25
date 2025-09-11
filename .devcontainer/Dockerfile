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

# Copy the project files (context is now parent directory)
COPY . /workspace/

# Clone the 2025_baseline_brain2speech repository
RUN mkdir -p /workspace/phase1/2025_baseline_brain2speech && \
    git clone https://github.com/LeonardoFerrisi/2025_baseline_brain2speech.git /workspace/phase1/2025_baseline_brain2speech

# Create a non-root user
RUN useradd -m -s /bin/bash vscode && \
    chown -R vscode:vscode /workspace

USER vscode

# Set working directory to phase1
WORKDIR /workspace/phase1

# Default command
CMD ["/bin/bash"]


