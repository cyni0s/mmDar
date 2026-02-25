FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /mmdar

# Install system dependencies (fixes broken install.sh)
RUN apt-get update && apt-get install -y python3-opencv && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
