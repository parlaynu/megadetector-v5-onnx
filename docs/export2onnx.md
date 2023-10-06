# Export to ONNX

This is reasonably complicated as there are a lot of dependencies to manage and some of them are getting
reasonably old.

This document explains how to run the export on a MacBook Pro x86_64 running MacOS 11.7.1.

NOTE: if you have a working docker environment, the below can be automated using the setup in the 'contrib' directory.

## Python 3.9

This requires `python3.9` to get all the python packages at the correct versions. It isn't on the Mac by default.
A simple way to get it is to use homebrew.

Follow the instructions at the [homebrew website](https://brew.sh/) to install the package manager.

Then install python3.9 with this command:

    brew install python@3.9

You will need to make sure `/usr/local/bin` is in your path.

## Create Workspace

Create the project workspace:

    mkdir -p ~/Projects/megadetector
    cd ~/Projects/megadetector
    export MEGAEXPORT=`pwd`

Clone this repository:

    cd $MEGAEXPORT
    git clone https://github.com/parlaynu/megadetector-v5-onnx

Clone yolo-v5 repository:

    cd $MEGAEXPORT
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    git checkout c23a441c9df7ca9b1f275e8c8719c949269160d1

Download the megadetector v5 saved models:

    cd $MEGAEXPORT
    mkdir models
    wget https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt -O models/md_v5a.0.0.pt
    wget https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt -O models/md_v5b.0.0.pt

## Python Environment Setup

Create a python virtual environment:

    cd $MEGAEXPORT
    python3.9 -m venv pyenv
    source pyenv/bin/activate

Install the versions of torch and torchvision needed by megadetector v5:

    pip install torch==1.10.2 torchvision==0.11.3

Install ONNX software:

    pip install onnx==1.12.0 onnxruntime==1.13.1 onnxoptimizer==0.3.2

YoloV5 has a long list of packages to install, however, they're not all needed for exporting. Here's
a minimal list to install:

    pip install pyyaml==6.0 requests==2.28.1 tqdm==4.64.1
    pip install opencv-python-headless==4.6.0.66 pandas==1.5.1 seaborn==0.12.1

## Run the Export

The exporter saves the exported model in the same directory as the original model but with a `onnx` extension. 
You need to specify the image size and batch size you need when exporting.

To export a dynamic model, the command is like this:

    cd $MEGAEXPORT/yolov5
    python3 export.py --include onnx \
                --weights <path-to-model.pt> \
                --dynamic

For example, exporting model-a for dynamic inputs:

    python3 export.py --include onnx \
                --weights ../models/md_v5a.0.0.pt \
                --dynamic

To export a static model, the command is like this:

    cd $MEGAEXPORT/yolov5
    python3 export.py --include onnx \
                --weights <path-to-model.pt> \
                --img-size <input-height> <input-width> --batch-size <batch-size>


## Optional: Optimize the Model

You can optionally use the `tools/optimize-onnx.py` script to optimize the model. It uses
[onnxoptimizer](https://github.com/onnx/optimizer) to generate an optimized version of the model.

It didn't seem to improve times very much in my testing.

Run the optimizer over the model with this command:

    cd $MEGAEXPORT/megadetector-v5-onnx
    ./tools/optimize-onnx.py ../models/md_v5a.0.0.onnx

The creates a model named `md_v5a.0.0_opt.onnx` in the same directory as the source model.
