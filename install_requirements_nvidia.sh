#!/bin/bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install diffusers==0.34.0
python -m pip install accelerate==0.31.0
python -m pip install "transformers>=4.25.1"
python -m pip install ftfy
python -m pip install Jinja2
python -m pip install datasets
python -m pip install wandb
python -m pip install onnxruntime-gpu==1.19.2
python -m pip install omegaconf
python -m pip install einops
python -m pip install torchmetrics
python -m pip install clean-fid
python -m pip install scikit-image
python -m pip install opencv-python
python -m pip install fvcore
python -m pip install cloudpickle
python -m pip install pycocotools
python -m pip install av
python -m pip install scipy
python -m pip install peft
pip install "huggingface-hub>=0.34.0,<1.0"  
