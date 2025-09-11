#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root or use sudo."
  exit 1
fi

# Update package list and install prerequisites
apt update && apt install -y software-properties-common

# Add deadsnakes PPA for Python 3.10
# Allows us to install Python 3.10 on Ubuntu versions that do not have it in their default repositories
add-apt-repository -y ppa:deadsnakes/ppa
apt update

echo "[Setup] Finished updating Linux packages"

# Install git LFS
apt-get install git-lfs

echo "[Setup] Finished installing git-lfs"

# Install Python 3.10
# apt install -y python3 python3-venv python3.10-dev python3-pip

# Verify Python installation
echo "[Setup] Getting Python Version"
python3 --version

echo "[Setup] Setting up Python environment"
# Create a virtual environment
python3 -m venv .b2txt2025

echo "[Setup] Activating Python environment and installing dependencies"
# Activate the virtual environment
source .b2txt2025/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "[Setup] Deactivating environment"
# Deactivate the virtual environment
deactivate

# Print completion message
echo "Setup complete. Python 3 and dependencies have been installed."