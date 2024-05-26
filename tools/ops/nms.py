from itertools import islice
import numpy as np


def non_max_suppression(preds, conf_thresh=0.25, iou_thresh=0.45, *, debug=False):
    
    # prediction: x, y, w, h, box_conf, cls0_conf, cls1_cnf, ...
    if debug:
        _print_preds("Raw Predictions", preds)
    
    # filter (by box conf_thresh) and sort the predictions
    preds = preds[preds[..., 4] > conf_thresh]
    preds = preds[np.flip(np.argsort(preds[..., 4], axis=-1), axis=0)]

    # replace class prob with class label
    preds[..., 5] = np.argmax(preds[..., 5:], axis=-1)
    preds = preds[..., :6]
    if debug:
        _print_preds("Preprocessed Predictions", preds)
    
    # convert boxes to xyxy
    preds[..., :4] = _xywh2xyxy(preds[..., :4])
    if debug:
        _print_preds("Coordinates Converted", preds)

    # offset boxes based on prediction
    preds, offset = _add_offset(preds)
    if debug:
        _print_preds("Offset Added", preds)
    
    # run the nms
    preds = _nms(preds, iou_thresh)
    if debug:
        _print_preds("NMS", preds)
    
    # remove the box offsets
    preds = _del_offset(preds, offset)
    if debug:
        _print_preds("Offset Removed", preds)
    
    # and done
    return preds


def _nms(preds, iou_thresh):
    
    p0 = preds[0]
    px = preds[1:]

    npreds = []
    npreds.append(p0)
    
    while len(px) > 0:
        # create the IOUs
        ious = _calc_ious(p0, px)
    
        # get the remaining predictions
        px = px[ious < iou_thresh]
        if len(px) == 0:
            break
        
        # prepare for next iteration
        p0 = px[0]
        px = px[1:]
        
        npreds.append(p0)

    return np.array(npreds)


def _calc_ious(b0, bx):
    # intersection area: (max(0, min(bottom) - max(top))) * (max(0, min(right) - max(left)))
    i_area = np.maximum(np.minimum(b0[2:4], bx[..., 2:4]) - np.maximum(b0[:2], bx[..., :2]), 0).prod(axis=1)
    
    # union area: area_b0 + area_bx - intersection area
    u_area = (b0[2:4] - b0[:2]).prod(axis=0) + (bx[..., 2:4] - bx[..., :2]).prod(axis=-1) - i_area
    
    return i_area / u_area


def _xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xc, yc, half_w, half_h = xywh[:, 0], xywh[:, 1], xywh[:, 2]/2, xywh[:, 3]/2
    xyxy[:, 0] = xc - half_w
    xyxy[:, 1] = yc - half_h
    xyxy[:, 2] = xc + half_w
    xyxy[:, 3] = yc + half_h
    return xyxy


def _add_offset(xyxy):
    offset = np.max(xyxy[..., 2]) + 1
    xyxy[..., 0] += offset*xyxy[..., 5]
    xyxy[..., 2] += offset*xyxy[..., 5]
    return xyxy, offset


def _del_offset(xyxy, offset):
    xyxy[..., 0] -= offset*xyxy[..., 5]
    xyxy[..., 2] -= offset*xyxy[..., 5]
    return xyxy


def _print_preds(desc, preds, top=10):
    print(f"{desc}: {len(preds)}")
    for idx in range(min(len(preds), top)):
        print([f"{p:0.4f}" for p in preds[idx]])


def _check_ious(preds, iou_thresh):
    for idx, pred0 in enumerate(preds[:-1]):
        ious = _calc_ious(pred0, preds[idx+1:])
        max_iou = np.max(ious)
        assert max_iou < iou_thresh


def nms_test():
    import time
    import random
    
    seed = 1234
    random.seed(seed)
    
    # definitions
    conf_thresh = 0.4
    iou_thresh = 0.4
    nclasses = 4
    npreds = 10000
    img_w, img_h = 1024, 768

    # create the predictions
    # xc, yc, w, h, conf, preds....
    preds = np.zeros((npreds, 5+nclasses))
    
    for idx in range(npreds):
        xc = 128 + random.randrange(0, 1024-256)
        yc = 128 + random.randrange(0, 768-256)
        width = random.randrange(64, 256)
        height = random.randrange(64, 256)
        conf = random.uniform(0.3, 0.9)
        kls = random.randrange(0, nclasses)
                
        preds[idx][0] = xc
        preds[idx][1] = yc
        preds[idx][2] = width
        preds[idx][3] = height
        preds[idx][4] = conf
        preds[idx][5+kls] = 1.0

    # run the algorithm
    start = time.monotonic()
    nms_preds = non_max_suppression(preds, conf_thresh, iou_thresh, debug=True)
    duration = time.monotonic() - start

    print(f"Processed {npreds} in {duration:0.4f} seconds")
    
    # some simple checks that only work if:
    #    seed = 1234
    #    npreds = 10000
    if seed != 1234 or npreds != 10000:
        return
    
    assert len(nms_preds) == 1119
    
    # check the iou thresholds for class 0
    kls_len = 0
    for kls in range(nclasses):
        kls_preds = nms_preds[nms_preds[..., 5] == float(kls)]
        kls_len += len(kls_preds)
        print(f"Class {kls} has {len(kls_preds)} predictions")
        _check_ious(kls_preds, iou_thresh)
        print("- iou test passed")
        
    assert kls_len == len(nms_preds)



if __name__ == "__main__":
    nms_test()

