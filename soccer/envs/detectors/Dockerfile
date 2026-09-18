FROM nvcr.io/nvidia/pytorch:26.01-py3

# Pin the existing GB10-compatible NVIDIA wheels: pip must fail, never replace them.
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')))"
COPY envs/detectors/requirements.txt /tmp/detectors.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-constraints.txt -r /tmp/detectors.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')); import rfdetr, ultralytics; print('RF-DETR', version('rfdetr'))"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 YOLO_CONFIG_DIR=/tmp/yolo RF_HOME=/models
WORKDIR /work

