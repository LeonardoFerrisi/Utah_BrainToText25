FROM python:3.13

# Set the working directory
WORKDIR /workspace

# Install git-lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Install the application dependencies
# Phase 1: copy local requirements file, Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    # pip install --no-cache-dir -r requirements.txt
    pip install -r requirements.txt


# Copy the rest of the application code
COPY . .
