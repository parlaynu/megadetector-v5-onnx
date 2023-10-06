#!/bin/sh

docker build -t megadetector-onnx .
docker run --rm -it -v "$PWD"/models:/megadetector/models megadetector-onnx
