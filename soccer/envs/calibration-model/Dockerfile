FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')))"
COPY envs/calibration-model/requirements.txt /tmp/calibration-model.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-constraints.txt -r /tmp/calibration-model.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')); import yaml, shapely, matplotlib, scipy, cv2, ultralytics; print('Calibration dependencies import successfully')"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    MPLBACKEND=Agg YOLO_CONFIG_DIR=/tmp/yolo
WORKDIR /work
