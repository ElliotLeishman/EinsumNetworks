#!/bin/bash

rm -rf venv/
virtualenv --system-site-packages -p python3.10 ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
python -m pip install torch
python -m pip install numpy
python -m pip install pillow
python -m pip install scipy
python -m pip install torch-fidelity
python -m pip install torchvision
python -m pip install torchmetrics
cd src
python datasets.py
