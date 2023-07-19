#!/usr/bin/env python3
import os
import argparse
import onnx, onnxoptimizer

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='path to model file', type=str, default=None)
args = parser.parse_args()

src_model = args.model_path
opt_model = os.path.splitext(src_model)[0] + "_opt.onnx"

print("loading model...", flush=True)
model = onnx.load(src_model)

print("checking model...", flush=True)
onnx.checker.check_model(model)

print("optimizing model...", flush=True)
passes = onnxoptimizer.get_fuse_and_elimination_passes()
passes.remove('nop')
model = onnxoptimizer.optimize(model, passes)

print("checking optimized model...", flush=True)
onnx.checker.check_model(model)

print("saving optimized model...", flush=True)
with open(opt_model, "wb") as f:
    f.write(model.SerializeToString())

