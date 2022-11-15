#  Notes on Specific Platforms

## Raspberry Pi with PiCamera2

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

