FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')))"
COPY envs/mesh-model/requirements.txt /tmp/mesh-model.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-constraints.txt -r /tmp/mesh-model.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')); import pytorch_lightning, roma, yacs, omegaconf, cv2; print('lightning', version('pytorch-lightning'))"
# MOMENTUM_ENABLED unset means upstream uses the TorchScript rig from the snapshot, not the mhr package.
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
WORKDIR /work
