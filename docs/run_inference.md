# Running Inference

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


