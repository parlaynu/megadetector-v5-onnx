#!/bin/sh

md_v5a=/megadetector/models/md_v5a.0.0.pt
md_v5b=/megadetector/models/md_v5b.0.0.pt

echo "Downloading megadetector models..."
if [ ! -f $md_v5a ]; then
      wget --progress=dot:giga https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt -O $md_v5a
fi
if [ ! -f $md_v5b ]; then
      wget --progress=dot:giga https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt -O $md_v5b
fi

echo "Converting megadetector models to onnx format..."
cd /megadetector/yolov5 || exit
python3 export.py \
      --include onnx \
      --weights $md_v5a \
      --dynamic

python3 export.py \
      --include onnx \
      --weights $md_v5b \
      --dynamic

echo "Created ./models/md_v5a.0.0.onnx and ./models/md_v5b.0.0.onnx"
