FROM soccerviz-detectors:20260907
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/nvidia-training-constraints.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision', 'rfdetr', 'transformers', 'supervision', 'numpy', 'opencv-python')))"
COPY envs/detector-training/requirements.txt /tmp/detector-training.txt
RUN python -m pip install --no-cache-dir -c /tmp/nvidia-training-constraints.txt -r /tmp/detector-training.txt
RUN python -c "from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/nvidia-training-constraints.txt').read_text() == ''.join(f'{p}=={version(p)}\n' for p in ('torch', 'torchvision', 'rfdetr', 'transformers', 'supervision', 'numpy', 'opencv-python')); from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer; print('training imports verified')"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 RF_HOME=/models WANDB_MODE=disabled
WORKDIR /work
