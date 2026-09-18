FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN python -m pip install --no-cache-dir \
    "numpy==2.1.0" "opencv-python==4.12.0.88" "av==16.1.0" \
    "ultralytics==8.4.143" "gdown==5.2.2" \
    "pandas==2.2.3" "pyarrow==23.0.1" "scikit-learn==1.6.1"
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 YOLO_CONFIG_DIR=/work/.yolo
WORKDIR /work
