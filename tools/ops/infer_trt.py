import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
    

from .nms import non_max_suppression


def infer_trt(pipe, args):

    # create the engine
    with open(args.model_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())

    # prepare the engine
    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * args.batch_size
        
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            h_input = np.empty(shape=[size], dtype=np.float32)
            # cuda.pagelocked_empty(shape=[size], dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
        else:
            h_output = cuda.pagelocked_empty(shape=[size], dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)
        
    bindings = [int(d_input), int(d_output)]
    
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    
    for item in pipe:
        h_input = item['input']
        
        # run the forward pass
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        # reshape the predictions into batches
        pred = np.reshape(h_output, (h_input.shape[0], -1, 8))
        
        # NMS returns a list of predictions... one tensor for each baatch entry
        # - 6 items per prediction: x1, y1, x2, y2, conf, cls

        preds = []
        for idx, p in enumerate(pred):
            p = non_max_suppression(p, conf_thresh, iou_thresh)
            print(f"- {idx:02d}: found {len(p)} objects")
            preds.append(p)

        item['pred'] = preds

        yield item

