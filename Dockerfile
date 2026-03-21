FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /mmdar

# Install system dependencies (fixes broken install.sh)
RUN apt-get update && apt-get install -y python3-opencv && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pytorch3d: point cloud ops for v2 training (chamfer_distance, knn_points)
# No PyPI wheel available — install from source at stable tag
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

COPY . .
