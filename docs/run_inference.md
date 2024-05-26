# Running Inference

It's very simple to run inference with the model once the export has happened.

From the root of this repository, run these three steps to setup the environment:

    python3.9 -m venv pyenv
    source pyenv/bin/activate
    pip install -r requirements.txt

There are three requirements files that can be used:

| requirements file         |                                                |
|---------------------------|------------------------------------------------|
| requirements.txt          | requirements for ONNX inference on CPUs        |
| requirements-gpu.txt      | requirements for ONNX inference on Nvidia GPUs |
| requirements-openvino.txt | requirements for ONNX inference using OpenVINO |

There are two tools for running inference:

| tool        |                                                 |
|-------------|-------------------------------------------------|
| run-onnx.py | run inference on ONNX models with onnxruntime   |
| run-trt.py  | run inference on TensorRT models using TensorRT |

They both work in the same general way. The function `build_pipeline` builds a pipeline of 
operations implemented as generator functions which is then iterated over.

## Image Source Specification

The `run-onnx.py` and `run-trt.py` tools have an `image_src` parameter which can be used to specify the source of images
for inference. This can be on disk image files, video files or directory hierarchies of image/video files, and 
camera devices.

The table below has the details.

| Value                         | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| path_to_directory             | load images from files matching default 'extensions'      |
| path_to_directory/...         | search full directory hierarchy for images                |
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

Any combination of prefix,hflip,vflip can be used with the camera devices.


## ONNX Inference

The `run-onnx.py` tool accepts the following arguments:

    usage: run-onnx.py [-h] [-c] [-p] [-x] [-a] [-t CONF_THRESH] [-u IOU_THRESH] [-n NUM_BATCHES] [-b BATCH_SIZE]
                       [-s IMAGE_SIZE]
                       model_path image_src [output_dir]
                       
    positional arguments:
      model_path            path to model file
      image_src             source of images - directory, file, or special
      output_dir            path to write output images
      
    optional arguments:
      -h, --help            show this help message and exit
      -c, --force-cpu       use the CPU even if there is an accelerator
      -p, --preserve-aspect
                            preserve image aspect ratio (pad if needed)
      -x, --cut-objects     cut detected objects from full image and save as individual images
      -a, --save-all        save all images, not just those with detections
      -t CONF_THRESH, --conf-thresh CONF_THRESH
                            confidence threshold for nms
      -u IOU_THRESH, --iou-thresh IOU_THRESH
                            iou threshold for nms
      -n NUM_BATCHES, --num-batches NUM_BATCHES
                            number of batches to process
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            batch size for dynamic model
      -s IMAGE_SIZE, --image-size IMAGE_SIZE
                            image <width>x<height> for dynamic model

For static models, the batch, width and height arguments can not be specified; for dynamic models, they must be
provided.

### Example

A simple example for using it to process 100 batches of images with a static model:

    ./tools/run-onnx.py -p -n 100 \
                    ../megamodels/md_v5a.0.0_640x512_1.onnx \
                    ~/Projects/datasets/fgvc8-iwildcam-2021/train/ \
                    ../outputs/
    preparing session
    - available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    - in use providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    building pipeline
    - input shape: 1 3 512 640
    - output shape: [1, 20400, 8]
    running
    000000 loading /home/paul/Projects/datasets/fgvc8-iwildcam-2021/train/86760c00-21bc-11ea-a13a-137349068a90.jpg
    - resizing from 1920x1080 to 640x360
    - padding from 640x360 to 640x512
    - 00: found 3 objects
    000001 loading /home/paul/Projects/datasets/fgvc8-iwildcam-2021/train/8676197a-21bc-11ea-a13a-137349068a90.jpg
    - resizing from 2048x1536 to 640x480
    - padding from 640x480 to 640x512
    - 00: found 1 objects
    ...
    ...
    ...
    summary
    - total runtime: 8.40
    -       average: 0.08

## TensorRT Inference

The `run-trt.py` tool is very similar to `run-onnx.py`. It currently has a limitation of only working
with a batch size of 1.

    usage: run-trt.py [-h] [-n NUM_BATCHES] [-a] [-p] [-x] [-t CONF_THRESH] [-u IOU_THRESH]
                      [-W WIDTH] [-H HEIGHT]
                      model_path image_src [output_dir]
                      
    positional arguments:
      model_path            path to model file
      image_src             source of images - directory, file, or special
      output_dir            path to write output images
      
    optional arguments:
      -h, --help            show this help message and exit
      -n NUM_BATCHES, --num-batches NUM_BATCHES
                            number of batches to process
      -a, --save-all        save all images, not just those with detections
      -p, --preserve-aspect
                            preserve image aspect ratio (pad if needed)
      -x, --cut-objects     cut detected objects from full image and save as individual images
      -t CONF_THRESH, --conf-thresh CONF_THRESH
                            confidence threshold for nms
      -u IOU_THRESH, --iou-thresh IOU_THRESH
                            iou threshold for nms
      -W WIDTH, --width WIDTH
                            processing width
      -H HEIGHT, --height HEIGHT
                            processing height


### Example

To run inference on 100 batches, with batch size of 1 and processing resolution of 640x512:

    ./tools/run-trt.py -p -n 100 -W 640 -H 512 \
            ../megamodels/md_v5a.0.0_640x512_1.trt \
            ~/Projects/datasets/fgvc8-iwildcam-2021/train/ \
            ../outputs/
    
    building pipeline
    running
    000000 loading /home/paul/Projects/datasets/fgvc8-iwildcam-2021/train/86760c00-21bc-11ea-a13a-137349068a90.jpg
    - resizing from 1920x1080 to 640x360
    - padding from 640x360 to 640x512
    - 00: found 3 objects
    000001 loading /home/paul/Projects/datasets/fgvc8-iwildcam-2021/train/8676197a-21bc-11ea-a13a-137349068a90.jpg
    - resizing from 2048x1536 to 640x480
    - padding from 640x480 to 640x512
    - 00: found 1 objects
    ...
    ...
    ...
    summary
    - total runtime: 7.63
    -       average: 0.08

