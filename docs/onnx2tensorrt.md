# Exporting ONNX to TensorRT - WIP

This is based on the information in the following documents:

* https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#ex-deploy-onnx
* https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/install_pycuda.sh

Note that you need to export the TensorRT model on the same machine you plan to run it on.

I started this because I was hoping I could run this on a Jetson Nano on the GPU, however, my
attempt to convert on that machine failed. I have run it successfully on a larger x86_64 machine.


## Install Dependencies

Both the `cuda` and `tensorrt` packages need to be installed. My system has cuda version 11.8.

    sudo apt install cuda-11-8
    sudo apt install tensorrt

These packages are required for building pycuda:

    sudo apt-get install -y build-essential python3-dev
    sudo apt-get install -y libboost-python-dev libboost-thread-dev

The python packages can be installed into a vitual environment or to the user or system packages. The
`tensorrt` package exists and is ready to install.

    pip install tensorrt

The `pycuda` package needs to be built:

    arch=$(uname -m)
    

## Convert The Model

Converting is as simple as:

    /usr/src/tensorrt/bin/trtexec --onnx=<model_name>.onnx --saveEngine=<model_name>.trt

