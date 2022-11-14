## Other Tools - WIP

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

