from .nms import non_max_suppression


def infer_onnx(pipe, sess, conf_thresh, iou_thresh):

    for item in pipe:
        inp = item['input']

        # run the forward pass
        print("- running inference", flush=True)
        pred = sess.run(None, {'images': inp})
        
        # NMS returns a list of predictions... one tensor for each batch entry
        # - 6 items per prediction: x1, y1, x2, y2, conf, cls
        
        preds = []
        for idx, p in enumerate(pred[0]):
            p = non_max_suppression(p, conf_thresh, iou_thresh)
            print(f"- {idx:02d}: found {len(p)} objects")
            preds.append(p)
            
        item['pred'] = preds

        yield item

