# Image Sources

The `run-onnx.py` tool has an `image_src` parameter which can be used to point to a directory on disk containint
images and/or videos or directly to an image or video file.

Additionally, there are some special keywords for this parameter that will cause it to read images from, for example,
a RaspberryPi camera.

The full details of the available options are listed in the table below.

| Value                         | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| path_to_directory             | load images from files matching default 'extensions'      |
| path_to_directory:ex1,ex2,... | load images from files matching specified 'extensions'    |
| path_to_file                  | load and process the single image                         |
| path_to_video                 | load frames from the video                                |
| path_to_video:start_frame     | seek start_frame position in video and then start reading |
| picamera2                     | read images from the RaspberryPi camera                   |
| picamera2:prefix              | save images with 'prefix' in name                         |
| picamera2:prefix,hflip,vflip  | optionally flip the image horizontal/vertical             |
| jetson_csi                    | read images from the Jetson CSI camera                    |
| jetson_csi:prefix             | save images with 'prefix' in name                         |
| jetson_csi:prefix,hflip,vflip | flip the image horizontal/vertical                        |

Any combination of prefix,hflip,vflip and be used with the camera devices.


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

