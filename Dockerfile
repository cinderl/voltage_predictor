FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code
COPY . /app

# default command runs the explain script
CMD ["python", "explain_model.py"]
# Use an official Python runtime as a parent image
# We choose a slim image for a smaller final image size
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
# This is often needed for some Python packages like numpy/pandas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
# This step is done early to leverage Docker's build cache
COPY requirements.txt /app/

# Install the Python dependencies
# We specify the library versions used in the model creation
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir matplotlib

# Copy the entire application code and data into the container
# The training script and the history.csv file are copied here
COPY . /app/

# Command to run the Python script when the container starts
# The script will execute the training process
CMD ["python", "train_model.py"]