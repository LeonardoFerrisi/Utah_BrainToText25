FROM python:3.13

# Set the working directory
WORKDIR /workspace

# Install the application dependencies
# Phase 1: copy local requirements file, Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
