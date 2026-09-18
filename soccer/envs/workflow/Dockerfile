FROM soccerviz-detectors:20260907
RUN python -c "from importlib.metadata import version; from pathlib import Path; Path('/tmp/vendor.txt').write_text(''.join(f'{p}=={version(p)}\n' for p in ('torch','torchvision','rfdetr','transformers','numpy')))"
RUN python -m pip install --no-cache-dir -c /tmp/vendor.txt \
    av==16.1.0 pyarrow==23.0.1 scikit-learn==1.6.1 \
    shapely==2.1.2 scipy==1.17.0 matplotlib==3.10.8 Pillow==12.1.1 \
    timm==1.0.22 albumentations==1.4.24 opencv-python-headless==4.12.0.88
COPY models/ /opt/soccerviz-models/
COPY upstreams/ /opt/soccerviz-upstreams/
RUN chmod -R a+rX /opt/soccerviz-models /opt/soccerviz-upstreams
RUN python -c "import av,cv2,pyarrow,timm,albumentations,rfdetr; from importlib.metadata import version; from pathlib import Path; assert Path('/tmp/vendor.txt').read_text()==''.join(f'{p}=={version(p)}\n' for p in ('torch','torchvision','rfdetr','transformers','numpy'))"
ENV PYTHONPATH=/work/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg YOLO_CONFIG_DIR=/tmp/yolo
WORKDIR /work
