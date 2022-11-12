# MegaDetector V5 - ONNX

This project explains how to export the [Megadetector V5](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) 
models to ONNX format, and provides some tools to easily run inference using the ONNX model.

The main reason I had for doing this was to make it easier to run on a range of platformss. In most cases, once you
have your model, just run a `pip install onnxruntime` (or onnxruntime-gpu or onnxruntime-openvine or ...) and with
the tools in this repository, you're ready to go. To make it easier, I've included an implementation of the 
non-maximum-suppression algorithm implemented in `numpy` so there's no need to for `pytorch` or any other library for that.

The exporting can be complicated, but you can export the model on any platform where all the requirements are readily 
satisfied, and then run inference on another with just the minimal dependencies.

One of the downsides is that you need to specify the image resolution and batch size at the conversion time; if you
want to change either, you need to re-export.

Additionally, not all platforms have prebuilt `onnxruntime` packages available. I've had to build from source for a few
platforms and it can be challenging. I'll add notes for some soon.

## Exporting to ONNX

This is reasonably complicated as there are a lot of dependencies to manage and some of them are getting
reasonably old.

The document linked to explains how to make it work on a RaspberryPi 4b 8G with a clean install of the
operating system. I documented it this way so I could capture all the software that was needed.

I also use it regularly on a MacbookPro x64 running MacOS 11.7.

Follow these [instructions](/docs/export2onnx.md).


## Running Inference

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

## Other Tools

### Model Checking

The tool `check-onnx.py` checks the model using the builtin checker from the onnx library.

    ./tools/check-onnx.py <path-to-model-file>

If the tool finds a problem, an exception is thrown.

### Model Information

The tool `model-info.py` displays information about the inputs and outputs from the model.

    ./tools/model-info.py <path-to-model-file>

An example output is:

    ./tools/model-info.py ../megamodels/md_v5a.0.0_640x512_1.onnx
    inputs
      00: name: images, shape: [1, 3, 512, 640], type: tensor(float)
    outputs
      00: name: output, shape: [1, 20400, 8], type: tensor(float)

### Model Optimizing

This tool optimizes the model using [onnxoptimizer](https://github.com/onnx/optimizer).

    ./tools/optimize-onnx.py <path-to-model-file>

It saves a version of the model in the same location as the source, but with `_opt` added to the file name.

I haven't seen much of a gain using this, but leaving it here just in case.


## References

* https://github.com/microsoft/CameraTraps/blob/main/megadetector.md
* https://github.com/ultralytics/yolov5

