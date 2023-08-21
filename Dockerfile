FROM python:3.9-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /megadetector
RUN pip3 install --no-cache-dir \
      opencv-python-headless==4.6.0.66 \
      onnx==1.12.0 \
      onnxruntime==1.13.1 \
      pandas==1.5.1 \
      pyyaml==6.0 \
      requests==2.28.1 \
      tqdm==4.64.1 \
      seaborn==0.12.1 \
      torch==1.9.0 \
      torchvision==0.10.0

WORKDIR /megadetector/yolov5
RUN git init . && \
    git remote add origin https://github.com/ultralytics/yolov5.git && \
    git fetch --depth 1 origin c23a441c9df7ca9b1f275e8c8719c949269160d1 && \ 
    git checkout FETCH_HEAD

WORKDIR /megadetector
COPY . .

CMD [ "./md2onnx.sh" ]
