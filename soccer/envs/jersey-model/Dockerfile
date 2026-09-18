FROM nvcr.io/nvidia/pytorch:26.01-py3

# Refuse replacement of the NVIDIA wheels required for GB10/sm_121.
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')))"
COPY envs/jersey-model/requirements.txt /tmp/jersey-model.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-constraints.txt -r /tmp/jersey-model.txt
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')); import timm, albumentations; print('timm', version('timm'))"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1
WORKDIR /work
