# Running Inference

It's very simple to run inference with the model once the export has happened.

From the root of this repository, run these three steps to setup the environment:

    python3.9 -m venv pyenv
    source pyenv/bin/activate
    pip install -r requirements.txt

To run on Nvidia GPUs using CUDA and/or TensorRT, install the from `requirements-gpu.txt`.
To run using OpenVINO, install from `requirements-openvino.txt`.

The tool `run-onnx.py` is used to run inference. It builds a pipeline of operations based on the
command line arguments passed in, and runs the pipeline. It uses generator functions to build
the pipeline.

I've tried to keep the code as simple as possible. If you can read python, it should be very easy
for you to understand what's going on. The `build_pipeline` function is where everything is put together.

Details on using the tool and the output are [here](/docs/run_inference.md).

The `image_src` parameter can take a number of different forms to specify local storage or camera devices.
See the [docs](/docs/image_sources.md) for details.

## Image Source Specification

The `run-onnx.py` and `run-trt.py` tools have an `image_src` parameter which can be used to specify the source of images
for inference. This can be on disk image files, video files or directory hierarchies of image/video files, and 
camera devices.

The table below has the details.

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


## Tool Overview

The `run-onnx.py` tool accepts the following arguments:

    usage: run-onnx.py [-h] [-n NUM_BATCHES] [-r] [-p] [-c] [-x] [-a] [-v FB_VIEW] model_path image_src [output_dir]

    positional arguments:
      model_path            path to model file
      image_src             source of images - directory, file, or special
      output_dir            path to write output images

    optional arguments:
      -h, --help            show this help message and exit
      -n NUM_BATCHES, --num-batches NUM_BATCHES
                            number of batches to process
      -r, --recurse         recursively search directory for images
      -p, --preserve-aspect
                            preserve image aspect ratio (pad if needed)
      -c, --force-cpu       use the CPU even if there is an accelerator
      -x, --cut-objects     cut detected objects from full image and save as individual images
      -a, --save-all        save all images, not just those with detections
      -v FB_VIEW, --fb-view FB_VIEW
                            display image on the framebuffer device

The input requirements of the model (image size, batch size) are automatically detected from the model. Input images are
resized and optionally padded (-p) as needed to match the size, and batches are constructed.

If an output directory is specified, images with detections are save there with bounding boxes around the detections. 
If '-c' is also specified, individual files for the crop areas are also saved. If '-a' is specified, all images are 
saved, not just those with detections.

For details on the `image_src` parameter, see [this](/docs/image_sources.md).

## Example

A simple example for using it to process 1 batch of images:

    ./tools/run-onnx.py -n 1 -p -x \
                    ../megamodels/md_v5a.0.0_640x512_1.onnx \
                    images/original \
                    images/output
    
    preparing session
    - available providers: ['CPUExecutionProvider']
    - in use providers: ['CPUExecutionProvider']
    building pipeline
    - input shape: [1, 3, 512, 640]
    - output shape: [1, 20400, 8]
    running
    0000 found image images/original/DSC04446.jpg
    - resizing from 2048x1365 to 640x426
    - padding from 640x426 to 640x512
    - processing image
    - 00: found 2 objects
    summary
    - total runtime: 1.67
    -  average step: 1.67


The output images are:

![bboxes](/images/processed/DSC04446_0000.jpg)
![cropped](/images/processed/DSC04446_0001.jpg)


