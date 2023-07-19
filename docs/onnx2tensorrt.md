# Exporting ONNX to TensorRT

NOTE: I have run this successfully on a larger x86_64 machine, however, there's no noticeable advantage
over using ONNX with the 'CUDAExecutionProvider'. I'd recommend skipping this unless you have a really
good reason to be using TensorRT.

This is based on the information in the following documents:

* https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#ex-deploy-onnx
* https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/install_pycuda.sh

Note that you need to export the TensorRT model on the same machine you plan to run it on.

I started this because I was hoping I could run this on a Jetson Nano on the GPU, however, that
device isn't able to convert the model.


## Install Dependencies

Both the `cuda` and `tensorrt` packages need to be installed. My system has cuda version 11.8.

    sudo apt install cuda-11-8
    sudo apt install tensorrt

These packages are required for building pycuda:

    sudo apt-get install -y build-essential python3-dev
    sudo apt-get install -y libboost-python-dev libboost-thread-dev

Install `tensorrt` python package:

    pip install --user tensorrt

The `pycuda` package needs to be built. Follow the instructions in the link below:

* https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/install_pycuda.sh

Once this is installed, you have the packages you need to run `run-trt.py`


## Convert The Model

Converting is as simple as:

    /usr/src/tensorrt/bin/trtexec --onnx=<model_name>.onnx --saveEngine=<model_name>.trt

