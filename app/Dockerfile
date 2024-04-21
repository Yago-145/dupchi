# Use a lightweight Ubuntu 22.04 base image
FROM python:3.11
#-slim-buster

# Install Python 3.11 and pip
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    rm -rf /var/lib/apt/lists/* 

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0

# Install the required packages
COPY *.whl .
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the application code and data to the container
COPY assets /app/assets
COPY model_checkpoints /app/model_checkpoints
COPY app_ui_gradio.py /app
COPY gcp /app/gcp

# Set the working directory to /app
WORKDIR /app

EXPOSE 8080

# Start the application
CMD ["python3", "app_ui_gradio.py"]
