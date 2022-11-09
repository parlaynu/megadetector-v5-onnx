# MegadetectorV5: Export to ONNX

## System Setup

Install the OS:

- follow these [instructions](https://www.raspberrypi.com/software/)
- Raspberry Pi OS Lite (64bit), 2022-09-22
- set the hostname, enable ssh, set the username password
- optionally set the ssh public key

Update the OS:

    sudo apt update
    sudo apt -y dist-upgrade
    sudo reboot

Install system packages:

    sudo apt install -y git
    sudo apt install -y python3-venv libpython3-dev


## Project Workspace

Create the project workspace.

    mkdir -p ~/Projects/megadetector
    cd ~/Projects/megadetector
    export PROJECTS=`pwd`

Clone this repository:

    cd $PROJECTS
    git clone https://github.com/parlaynu/megadetectorv5-onnx

Download the megadetector v5 saved models:

    cd $PROJECTS
    mkdir megamodels
    wget https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt -O megamodels/md_v5a.0.0.pt
    wget https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt -O megamodels/md_v5b.0.0.pt


## Software Installation

Setup python virtual environment:

    cd $PROJECTS
    python3.9 -m venv pyenv-export
    source pyenv-export/bin/activate

Install the versions of torch and torchvision needed by megadetector v5:

    pip install torch==1.10.2 torchvision==0.11.3

Install ONNX software:

    pip install onnx==1.12.0

YoloV5 has a long list of packages to install, however, they're not all needed for exporting. Here's
a minimal list to install:

    pip install pyyaml==6.0 requests==2.28.1 tqdm==4.64.1
    pip install opencv-python-headless==4.6.0.66 pandas==1.5.1 seaborn==0.12.1

## Install YoloV5 and Export

Clone yolo-v5 repository and install dependencies:

    cd $PROJECTS
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    git checkout c23a441c9df7ca9b1f275e8c8719c949269160d1

Run the exporter. On the Raspberry Pi, it prints out the following warning, but it doesn't impact
the running and it's safe to ignore it:

    UserWarning: Failed to load image Python extension

The exporter saves the exported model in the same directory as the original model but with a `onnx` extension. 
You need to specify the image size and batch size you need when exporting.

The general form of the exporter command is:

    python3 export.py --include onnx \
                --weights <path-to-model.pt> \
                --img-size <input-height> <input-width> --batch-size <batch-size>

For example, exporting model-a for inputs image resolution 1280x1024 and batch size 1:

    python3 export.py --include onnx \
                --weights ../megamodels/md_v5a.0.0.pt \
                --img-size 1024 1280 --batch-size 1


## Optional: Optimize the Model

You can optionally use the `tools/optimize-onnx.py` script to optimize the model. It didn't seem to 
improve times very much though.

This package needs to build from source for the Raspberry Pi. Seems to run quite smoothly though.

    pip install onnxoptimizer

Run the optimizer over the model. The example below will create a model named `md_v5a.0.0-opt.onnx` in the
same directory as the source model.

    cd $PROJECTS/megadetectorv5-onnx/tools
    ./optimize-onnx.py ../../megamodels/md_v5a.0.0.onnx

