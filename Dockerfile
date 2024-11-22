# Use a specific PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL authors="hendrik"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    ffmpeg \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install  -r requirements.txt \
    gymnasium[box2d] \
    wandb

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the training script when the container launches
ENTRYPOINT ["python", "main.py"]
