#!/usr/bin/env python3
import sys, os
import time
from itertools import islice, count

import onnxruntime as ort



def prepare_session(model_path, force_cpu):
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


def build_pipeline(session, image_dir, args):
    import ops
    
    print("building pipeline")
    print(f"- input shape: {session.get_inputs()[0].shape}")
    print(f"- output shape: {session.get_outputs()[0].shape}")
    
    batch_size, nchans, height, width = session.get_inputs()[0].shape
    
    pipe = ops.find_images(image_dir, args.recurse, args.extensions)
    pipe = ops.load_image(pipe, width, height, nchans, args.preserve_aspect)

    if batch_size > 1:
        pipe = ops.batcher(pipe, batch_size)

    pipe = ops.transform_image(pipe)
    pipe = ops.infer(pipe, session)

    if args.output_dir is not None:
        pipe = ops.draw_bboxes(pipe)
        if args.crop_outputs:
            pipe = ops.crop_detections(pipe)

        pipe = ops.save_images(pipe, args.image_dir, args.output_dir, args.save_all)

    return pipe


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-batches', help='number of batches to process', type=int, default=0)
    parser.add_argument('-e', '--extensions', help='file extensions to identify as images (case insensitive)', type=str, default='jpg,jpeg')
    parser.add_argument('-r', '--recurse', help='recursively search directory for images', action='store_true')
    parser.add_argument('-p', '--preserve-aspect', help='preserve image aspect ratio (pad if needed)', action='store_true')
    parser.add_argument('-f', '--force-cpu', help='use the CPU even if there is an accelerator', action='store_true')
    parser.add_argument('-c', '--crop-outputs', help='crop the output into smaller images and save in output dir', action='store_true')
    parser.add_argument('-a', '--save-all', help='save all images, not just those with detections', action='store_true')
    parser.add_argument('model_path', help='path to model file', type=str, default=None)
    parser.add_argument('image_dir', help='path to images', type=str, default=None)
    parser.add_argument('output_dir', help='path to write output images', nargs='?', type=str, default=None)
    args = parser.parse_args()

    sess = prepare_session(args.model_path, args.force_cpu)
    pipe = build_pipeline(sess, args.image_dir, args)

    start = time.time()
    
    print("running")
    pipe = pipe if args.num_batches == 0 else islice(pipe, args.num_batches)
    for idx, item in enumerate(pipe):
        pass
    
    duration = time.time() - start
    average = duration / (idx+1)
    
    print("summary")
    print(f"- total runtime: {duration:0.2f}")
    print(f"-  average step: {average:0.2f}")


if __name__ == "__main__":
    main()

