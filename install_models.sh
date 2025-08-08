#!/bin/bash
git lfs install
mkdir pretrained_models
cd ./pretrained_models
GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
cd stable-diffusion-xl-base-1.0
git lfs pull
cd ../
