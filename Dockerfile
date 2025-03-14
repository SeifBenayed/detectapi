# Use CUDA-enabled Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /app/cache

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/cache

# Run the application
CMD ["uvicorn", "hemden:app", "--host", "0.0.0.0", "--port", "8000"]