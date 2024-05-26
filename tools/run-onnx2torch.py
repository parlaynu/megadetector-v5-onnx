#!/usr/bin/env python3
import sys, os
import time
from itertools import islice, count


def prepare_model(args):
    import torch
    from onnx2torch import convert
    
    device = torch.device('cuda') if (not args.force_cpu and torch.cuda.is_available()) else torch.device('cpu')
    print(f"running on {device}")

    print("- converting model to torch")
    model = convert(args.model_path)
    
    if args.fuse_layers:
        print(f"- fusing model")
        model = model.fuse().eval()
    else:
        model = model.eval()
    
    model = model.to(device)
    model.device = device

    return model


def build_pipeline(args, model):
    import ops
    
    print("building pipeline")

    batch_size, nchans, height, width = args.batch_size, 3, args.height, args.width
    print(f"- input shape: {batch_size} {nchans} {height} {width}")
    
    if args.image_src.startswith("picamera2"):
        params = []
        cidx = args.image_src.find(':', 2)
        if cidx != -1:
            params = args.image_src[cidx+1:].split(',')
        args.image_src = None
        pipe = ops.load_from_picamera2(params, width, height)
    
    elif args.image_src.startswith("jetson_csi"):
        params = []
        cidx = args.image_src.find(':', 2)
        if cidx != -1:
            params = args.image_src[cidx+1:].split(',')
        args.image_src = None
        pipe = ops.load_from_jetson_csi(params, width, height)
    
    else:  # fall back to images/videos from disk
        params = ['jpg', 'jpeg']
        cidx = args.image_src.find(':', 2)
        if cidx != -1:
            params = args.image_src[cidx+1:].split(',')
            args.image_src = args.image_src[:cidx]
        
        pipe = ops.load_images(args.image_src, params)

    if batch_size > 1:
        pipe = ops.batcher(pipe, batch_size)

    pipe = ops.transform_images(pipe, width, height, nchans, args.preserve_aspect)
    pipe = ops.infer_torch(pipe, model, args.conf_thresh, args.iou_thresh)

    if args.output_dir is not None:
        pipe = ops.draw_bboxes(pipe)
        if args.cut_objects:
            pipe = ops.cut_objects(pipe)


        if args.image_src is None:
            src_dir = None
        else:
            src_dir = args.image_src if os.path.isdir(args.image_src) else os.path.dirname(args.image_src)

        pipe = ops.save_images(pipe, src_dir, args.output_dir, args.save_all)
    
    return pipe


def main():
    import argparse, re
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--force-cpu', help='use the CPU even if there is an accelerator', action='store_true')
    parser.add_argument('-f', '--fuse-layers', help='fuse model layers', action='store_true')
    parser.add_argument('-p', '--preserve-aspect', help='preserve image aspect ratio (pad if needed)', action='store_true')
    parser.add_argument('-x', '--cut-objects', help='cut detected objects from full image and save as individual images', action='store_true')
    parser.add_argument('-a', '--save-all', help='save all images, not just those with detections', action='store_true')
    parser.add_argument('-t', '--conf-thresh', help='confidence threshold for nms', type=float, default=0.25)
    parser.add_argument('-u', '--iou-thresh', help='iou threshold for nms', type=float, default=0.45)
    parser.add_argument('-n', '--num-batches', help='number of batches to process', type=int, default=0)
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=1)
    parser.add_argument('-s', '--image-size', help='image <width>x<height> for processing', type=str, default="640x512")
    parser.add_argument('model_path', help='path to model file', type=str, default=None)
    parser.add_argument('image_src', help='source of images - directory, file, or special', type=str, default=None)
    parser.add_argument('output_dir', help='path to write output images', nargs='?', type=str, default=None)
    args = parser.parse_args()

    # process the args image size specifier
    size_re = re.compile(r"^(\d+)x(\d+)$")
    m = size_re.match(args.image_size)
    if m is None:
        print(f"Error: unrecognized image size specification: {args.image_size}")
        return None

    args.width = int(m.group(1))
    args.height = int(m.group(2))
    
    if args.width % 64 != 0 or args.height % 64 != 0:
        print("Error: megadetector requires width and height to be integer multiples of 64")
        return None

    # build the pipeline
    model = prepare_model(args)
    pipe = build_pipeline(args, model)
    if pipe is None:
        return

    start = time.time()
    
    print("running")
    pipe = pipe if args.num_batches == 0 else islice(pipe, args.num_batches)
    for idx, item in enumerate(pipe):
        # don't count the first iteration in the total time...
        #    can take a while in some instances
        if idx == 0:
            start = time.time()
        pass
    
    print("summary")

    duration = time.time() - start
    print(f"- total runtime: {duration:0.2f}")

    try:
        average = duration / (idx)
        print(f"-       average: {average:0.2f}")
    except UnboundLocalError:
        pass
        


if __name__ == "__main__":
    main()

