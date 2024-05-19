# Stage 1: Prepare the environment
FROM python:3.10 AS base

# Set environment variables
ENV DOCKER_CACHE_SIZE=50GB

# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# Stage 2: Copy application files
FROM base AS app

# Create directories
RUN mkdir -p /app/applications /app/shared /app/static /app/templates

# Copy application files
COPY applications /app/applications
COPY shared /app/shared
COPY static /app/static
COPY templates /app/templates
COPY app.py /app/app.py

# Set the working directory
WORKDIR /app

# Set metadata labels
LABEL version="1.0"
LABEL maintainer="mohsen_cnstart"

# Expose port
EXPOSE 5000

# Specify the command to run on container start
CMD ["python", "app.py"]
