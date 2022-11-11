#!/usr/bin/env python3
import sys, os
import time
from itertools import islice, count


def prepare_session(model_path, force_cpu):
    import onnxruntime as ort
    from onnxruntime.capi._pybind_state import get_available_providers
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    
    print(f"preparing session", flush=True)
    providers = get_available_providers()
    print(f"- available providers: {providers}")

    if force_cpu:
        providers = ['CPUExecutionProvider']
    
    sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
    print(f"- in use providers: {sess.get_providers()}")
    
    return sess


def build_pipeline(session, args):
    import ops
    
    print("building pipeline")
    print(f"- input shape: {session.get_inputs()[0].shape}")
    print(f"- output shape: {session.get_outputs()[0].shape}")
    
    batch_size, nchans, height, width = session.get_inputs()[0].shape
    
    if args.image_src.startswith("picamera2"):
        params = []
        cidx = args.image_src.find(':', 2)
        if cidx != -1:
            params = args.image_src[cidx+1:].split(',')
        pipe = ops.load_from_picamera2(params, width, height)
    
    elif args.image_src.startswith("jetson_csi"):
        params = []
        cidx = args.image_src.find(':', 2)
        if cidx != -1:
            params = args.image_src[cidx+1:].split(',')
        pipe = ops.load_from_jetson_csi(params, width, height)
    
    else:  # fall back to images/videos from disk
        image_src = args.image_src
        params = ['jpg', 'jpeg']
        cidx = args.image_src.find(':', 2)
        if cidx != -1:
            image_src = args.image_src[:cidx]
            params = args.image_src[cidx+1:].split(',')
        
        pipe = ops.load_images(image_src, params, args.recurse)

    pipe = ops.transform_images(pipe, width, height, nchans, args.preserve_aspect)

    if batch_size > 1:
        pipe = ops.batcher(pipe, batch_size)

    pipe = ops.infer(pipe, session)

    if args.output_dir is not None:
        pipe = ops.draw_bboxes(pipe)
        if args.cut_objects:
            pipe = ops.cut_objects(pipe)

        src_dir = args.image_src if os.path.isdir(args.image_src) else os.path.dirname(args.image_src)

        pipe = ops.save_images(pipe, src_dir, args.output_dir, args.save_all)
    
    return pipe


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-batches', help='number of batches to process', type=int, default=0)
    parser.add_argument('-r', '--recurse', help='recursively search directory for images', action='store_true')
    parser.add_argument('-p', '--preserve-aspect', help='preserve image aspect ratio (pad if needed)', action='store_true')
    parser.add_argument('-c', '--force-cpu', help='use the CPU even if there is an accelerator', action='store_true')
    parser.add_argument('-x', '--cut-objects', help='cut detected objects from full image and save as individual images', action='store_true')
    parser.add_argument('-a', '--save-all', help='save all images, not just those with detections', action='store_true')
    parser.add_argument('model_path', help='path to model file', type=str, default=None)
    parser.add_argument('image_src', help='source of images - directory, file, or special', type=str, default=None)
    parser.add_argument('output_dir', help='path to write output images', nargs='?', type=str, default=None)
    args = parser.parse_args()

    sess = prepare_session(args.model_path, args.force_cpu)
    pipe = build_pipeline(sess, args)

    start = time.time()
    
    print("running")
    pipe = pipe if args.num_batches == 0 else islice(pipe, args.num_batches)
    for idx, item in enumerate(pipe):
        pass
    
    print("summary")

    duration = time.time() - start
    print(f"- total runtime: {duration:0.2f}")

    try:
        average = duration / (idx+1)
        print(f"-       average: {average:0.2f}")
    except UnboundLocalError:
        pass
        


if __name__ == "__main__":
    main()

