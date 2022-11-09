# tail-recursive implementation of NMS
import numpy as np


def non_max_suppression(pred, conf_thresh=0.25, iou_thresh=0.45):
    
    # prediction: x, y, w, h, box_conf, cls0_conf, cls2_cnf, ...
    num_classes = pred.shape[1] - 5
    
    # filter and sort the predictions by confidence threshold
    pred = pred[pred[..., 4] > conf_thresh]
    pred = pred[np.flip(np.argsort(pred[..., 4], axis=-1))]

    # replace class prob with class label
    pred[..., 5] = np.argmax(pred[..., 5:], axis=-1)
    pred = pred[..., :6]
    
    # convert boxes to xyxy
    pred[..., :4] = xywh2xyxy(pred[..., :4])

    # run the nms
    return _nms(pred, conf_thresh, iou_thresh, [])


def _nms(pred, conf_thresh, iou_thresh, npred):
    
    if len(pred) == 0:
        return npred

    p0 = pred[0]
    px = pred[1:]

    npred.append(p0)
    if len(px) == 0:
        return npred
    
    # create the IOUs
    ious = calc_ious(p0, px)
    
    # for non-matching class, set the iou to 0 so they don't get considered
    ious[px[..., 5] != p0[5]] = 0

    # get the remaining predictions
    pp = px[ious < iou_thresh]

    return _nms(pp, conf_thresh, iou_thresh, npred)


def calc_ious(b0, bx):
    # intersection area: (max(0, min(bottom) - max(top))) * (max(0, min(right) - max(left)))
    i_area = np.maximum(np.minimum(b0[2:4], bx[..., 2:4]) - np.maximum(b0[:2], bx[..., :2]), 0).prod(axis=1)
    
    # union area: area_b0 + area_bx - intersection area
    u_area = (b0[2:4] - b0[:2]).prod(axis=0) + (bx[..., 2:4] - bx[..., :2]).prod(axis=-1) - i_area
    
    return i_area / u_area
    

def xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xc, yc, half_w, half_h = xywh[:, 0], xywh[:, 1], xywh[:, 2]/2, xywh[:, 3]/2
    xyxy[:, 0] = xc - half_w
    xyxy[:, 1] = yc - half_h
    xyxy[:, 2] = xc + half_w
    xyxy[:, 3] = yc + half_h
    return xyxy
