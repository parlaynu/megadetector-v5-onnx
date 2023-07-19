# Jetson Nano

The Jetson Nano has python3.6. It can't be update to 3.9 because a lot of the packages needed to use the jetson
are custom built by Nvidia to use the hardware. This includes `opencv` which is installed as part of the jetpack.

The installed jetpack I built from was 4.6.2:

    apt search jetpack | grep installed
    nvidia-jetpack/stable,now 4.6.2-b5 arm64 [installed]

## Timing Tests

Results from some testing.

* Image source: local disk
* Number of Images: 55
* Processing Resolution: 640x512
* Batch Size: 1

| Model/Options      | Per Image Time       |
|--------------------|----------------------|
| ONNX, CPU          |   7.70s              |
| Torch, CPU         |   8.35s              |
| Torch, CPU, Fused  |   8.20s              |
| Torch, GPU         |   1.10s              |
| Torch, GPU, Fused  |   1.11s              |


## Dependencies

Get an updated pip3 to install. Mainly because the system pip3 uses `--ignore-installed` by default which
continually wants to reinstall packages, overriding the system installed packages and breaking things.

    pip3 install --user --upgrade pip

You will probably need a fresh login after installing pip - paths seem to be getting confused.

An updated wheel package is also needed to build the wheel for onnxruntime:

    pip3 install --user --upgrade wheel
    pip3 install --user packaging
    pip3 install --user cython

I think that gcc-8 and g++-8 are needed... not 100% sure, but I've been using it and I think
this was the reason I installed it.

    sudo apt install gcc-8 g++-8

An updated version of cmake is also needed. The easiest way to get one is to download a binary build from 
kitware's github releases.

    wget https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-linux-aarch64.sh
    sh cmake-3.24.3-linux-aarch64.sh

Make sure the `bin` directory of the installation is on your path.

Other dependencies:

    sudo apt install build-essential git cmake ninja-build
    sudo apt install libpython3-dev
    sudo apt install libomp5 libomp-dev libopenblas-dev
    sudo apt install libpng-dev libjpeg-dev

Other python packages

    pip3 install pyyaml
    pip3 install pillow


## Protobuf

Instructions can be found here:

    https://github.com/onnx/onnx/tree/rel-1.10.0#building-protobuf-from-source

    export CC=/usr/bin/gcc-8
    export CXX=/usr/bin/g++-8

    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git checkout v3.16.0
    git submodule update --init --recursive
    mkdir build_source && cd build_source
    cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install


## Numpy

Version 1.19.5 is the latest version that attempts to install from a wheel using pip3. However, importing it fails with
an "Illegal Instruction" error. 

    export CC=/usr/bin/gcc-8
    export CXX=/usr/bin/g++-8

    git clone https://github.com/numpy/numpy.git
    cd numpy
    git checkout v1.19.5
    git submodule update --init --recursive

    python3 setup.py bdist_wheel

    pip3 install --user dist/numpy-1.19.5-cp36-cp36m-linux_aarch64.whl


## Onnx

Instructions can be found here:

    https://github.com/onnx/onnx/tree/rel-1.10.0#build-onnx-from-source

    export CC=/usr/bin/gcc-8
    export CXX=/usr/bin/g++-8

    git clone https://github.com/onnx/onnx.git
    cd onnx
    git checkout v1.12.0
    git submodule update --init --recursive
    # prefer lite proto
    export CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
    pip3 install --user .


## Onnxruntime

The build reference is from here:

    https://onnxruntime.ai/docs/build/inferencing.html

Build onnxruntime:

    export CC=/usr/bin/gcc-8
    export CXX=/usr/bin/g++-8

    git clone https://github.com/microsoft/onnxruntime.git
    cd onnxruntime
    git checkout v1.13.1
    git submodule sync
    git submodule update --init --recursive --jobs 1

    ./build.sh --config Release --build --update --parallel --enable_pybind --build_wheel --skip_tests

    cd build/Linux/Release/dist
    pip3 install --user *whl


## Torch and Torchvision with CUDA Support

The torch wheels available don't support cuda, so it needs to be built from source. 

However, there's something missing either in the build instructions below, or the jetson environment - the 
inference works when run on the GPU, but running on the CPU fails. Using the standard `pip` installed
torch and torchvision works fine for the CPU.

I've tried various things, but haven't been able to resolve the problem.

Build torch 1.10.2

    pip3 install astunparse ninja cffi typing_extensions future six requests dataclasses

    export MAX_JOBS=4
    export USE_CUDA=1
    export USE_OPENCV=1
    export BLAS=OpenBLAS
    export BUILD_TEST=0

    git clone https://github.com/pytorch/pytorch torch
    cd torch
    git checkout v1.10.2
    git submodule sync
    git submodule update --init --recursive --jobs 1

    python3 setup.py bdist_wheel

    pip install dist/*whl

Build torchvision 0.11.3

    git clone https://github.com/pytorch/vision.git torchvision
    git checkout v0.11.3

    python3 setup.py bdist_wheel

    pip install dist/*whl

