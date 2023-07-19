import torch
from .nms import non_max_suppression


def infer_torch(pipe, model, conf_thresh, iou_thresh):

    device = model.device
    for item in pipe:
        inp = torch.from_numpy(item['input'])
        inp = inp.to(device)
        
        # run the forward pass
        with torch.no_grad():
            pred = model(inp)
        
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.detach().cpu().numpy()
        
        # NMS returns a list of predictions... one tensor for each baatch entry
        # - 6 items per prediction: x1, y1, x2, y2, conf, cls
        
        preds = []
        for idx, p in enumerate(pred):
            p = non_max_suppression(p, conf_thresh, iou_thresh)
            print(f"- {idx:02d}: found {len(p)} objects", flush=True)
            preds.append(p)
            
        item['pred'] = preds

        yield item

