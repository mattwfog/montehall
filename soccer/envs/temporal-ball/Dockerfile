FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')))"
COPY envs/temporal-ball/requirements.txt /tmp/temporal-ball.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-constraints.txt -r /tmp/temporal-ball.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')); import cv2, hydra, torchvision"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4
WORKDIR /work

