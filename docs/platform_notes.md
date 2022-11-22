#  Notes on Specific Platforms

## Raspberry Pi

### PiCamera2

The PiCamera2 complicates the installation a little - it is available on pypi.org however, not all of
its dependencies are there. The only way to install it is using a system level package, so that means
virtual environments can't be used.

The next best option to avoid installing packages at the system level is to do a user level install. 
This can be done with the following:

    sudo apt install -y python3-picamera2 python3-pip
    pip3 install --user -r requirements.txt

To use the camera, run the tool with a command line like:

    ./tools/run-onnx.py -n 3 -p -c ../megamodels/md_v5a.0.0_640x512_1.onnx \
                    picamera2:mycamera \
                    ../output

The resulting images will be named like this:

    -rw-r--r-- 1 pi pi 82514 Nov 10 12:50 mycamera_000000_0000.jpg
    -rw-r--r-- 1 pi pi 86705 Nov 10 12:50 mycamera_000001_0000.jpg
    -rw-r--r-- 1 pi pi 89641 Nov 10 12:50 mycamera_000002_0000.jpg
    -rw-r--r-- 1 pi pi 63177 Nov 10 12:50 mycamera_000002_0001.jpg

The general form is:

    <prefix>_<image_id>_0000.jpg - this is the full image with bounding boxes
    <prefix>_<image_id>_XXXX.jpg - these are the crops taken from the image

## Jetson Nano

The Jetson Nano has python3.6. It can't update to 3.9 because a lot of the packages needed to use the jetson
are custom built by Nvidia to use the hardware. This includes `numpy` and `opencv` which are both installed
as part of the jetpack.

The package `onnxruntime` needs to be built from source. Here's the reference:

    https://onnxruntime.ai/docs/build/inferencing.html

The installed jetpack I built from was 4.6.2:

    apt search jetpack | grep installed
    nvidia-jetpack/stable,now 4.6.2-b5 arm64 [installed]

### Dependencies

Get an updated pip3 to install. Mainly because the system pip3 uses `--ignore-installed` by default which
breaks things.

    pip3 install --user --upgrade pip

You will probably need a fresh shell after installing pip - paths seem to be getting confused.

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


### Protobuf

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


### Numpy

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


### Onnx

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

### Onnxruntime

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

