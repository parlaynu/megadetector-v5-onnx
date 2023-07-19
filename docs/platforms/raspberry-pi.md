# Raspberry Pi

OS Version: RaspberryPi OS Lite (64 bit) 2022-09-22

## Timing Tests

Results from some testing on a RaspberryPi 4 with 8GBytes of RAM.

* Image source: local disk
* Number of Images: 55
* Processing Resolution: 640x512
* Batch Size: 1

| Model/Options      | Per Image Time       |
|--------------------|----------------------|
| ONNX, NCS2         |   1.51s              |   
| ONNX, CPU          |   6.67s              |
| Torch, CPU         |   8.24s              |
| Torch, CPU, Fused  |   8.26s              |

Where NCS2 is Intel's Neural Compute Stick 2.

## PiCamera2

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

## Onnxruntime with OpenVino

### Build Compiler

Version 11.3.0.

Download the source

    wget ftp://ftp.fu-berlin.de/unix/languages/gcc/releases/gcc-11.3.0/gcc-11.3.0.tar.xz
    tar xf gcc-11.3.0.tar.xz
    rm -f gcc-11.3.0.tar.xz
    cd gcc-11.3.0
    ./contrib/download_prerequisites

Configure

    mkdir -p build && cd build
    ../configure --enable-languages=c,c++ --with-cpu=cortex-a72

Build and Install

    make -j $(nproc)
    sudo make install

To Use:

    export CC=/usr/local/bin/gcc
    export CXX=/usr/local/bin/g++
    export LD_LIBRARY_PATH=/usr/local/lib64


### Build Environment

    sudo apt install build-essential git cmake libusb-1.0-0-dev
    sudo apt install opencl-dev

    sudo apt install python3-venv libpython3-dev

    python3 -m venv pyenv
    source pyenv/bin/activate
    pip3 install wheel==0.37.1 packaging
    pip3 install cython patchelf
    pip3 install numpy

    export CC=/usr/local/bin/gcc
    export CXX=/usr/local/bin/g++
    export LD_LIBRARY_PATH=/usr/local/lib64


### Build OpenVino

    git clone https://github.com/openvinotoolkit/openvino.git
    cd openvino
    git checkout 2022.2.0
    git submodule sync
    git submodule update --init --recursive --jobs 2

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/openvino \
        -DTHREADING=SEQ \
        -DENABLE_PYTHON=ON \
        -DPYTHON_EXECUTABLE=`which python3.9` \
        -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.9.so \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.9 \
        -DENABLE_WHEEL=ON \
        ..

    make -j $(nproc)
    sudo make install


### Build OnnxRuntime

    git clone https://github.com/Microsoft/onnxruntime
    cd onnxruntime
    git checkout v1.13.1
    git submodule sync
    git submodule update --init --recursive --jobs 1

    source /usr/local/openvino/setupvars.sh

    ./build.sh --config Release --build --update --parallel --enable_pybind --build_wheel \
                    --skip_tests --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
                    --use_openvino MYRIAD_FP16

    pip3 install dist/*whl

## USB Setup for Neural Compute Stick 2

    cat <<EOF > 97-myriad-usbboot.rules
    SUBSYSTEM=="usb", ATTRS{idProduct}=="2485", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
    SUBSYSTEM=="usb", ATTRS{idProduct}=="f63b", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
    EOF

    sudo cp 97-myriad-usbboot.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo ldconfig

    rm 97-myriad-usbboot.rules


## References

Building the Compiler

* https://forums.raspberrypi.com/viewtopic.php?t=310833
* https://forums.raspberrypi.com/viewtopic.php?t=310833#p1865571

Building OpenVINO

* https://onnxruntime.ai/docs/build/eps.html#openvino
* https://github.com/openvinotoolkit/openvino/wiki/BuildingForRaspbianStretchOS

Onnxruntime

* https://onnxruntime.ai/docs/build/eps.html#armnn
* https://onnxruntime.ai/docs/build/inferencing.html#native-compiling-on-linux-arm-device

