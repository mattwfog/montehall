FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')))"
COPY envs/pose-model/requirements.txt /tmp/pose-model.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-constraints.txt -r /tmp/pose-model.txt
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision')); import transformers, cv2; print('transformers', version('transformers'))"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
WORKDIR /work
