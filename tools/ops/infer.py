from .nms import non_max_suppression


def infer(pipe, sess):

    detection_threshold = 0.005
    for item in pipe:
        print(f"- processing image")
        inp = item['input']

        # run the forward pass
        pred = sess.run(None, {'images': inp})

        # NMS returns a list of predictions... one tensor for each baatch entry
        # - 6 items per prediction: x1, y1, x2, y2, conf, cls
        
        preds = []
        for idx, p in enumerate(pred[0]):
            p = non_max_suppression(p)
            print(f"- {idx:02d}: found {len(p)} objects")
            preds.append(p)
            
        item['pred'] = preds

        yield item

