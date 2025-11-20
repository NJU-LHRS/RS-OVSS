#!/bin/bash
set -e

pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    torchvision==0.16.2

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install mmengine==0.10.4 \
            mmsegmentation==1.2.2 \
            numpy==1.24.1 \
            opencv-python==4.6.0.66 \
            opencv-python-headless==4.7.0.72 \
            setuptools==60.2.0 \
            ftfy \
            regex \
            tqdm

