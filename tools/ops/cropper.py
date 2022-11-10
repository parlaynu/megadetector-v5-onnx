

def crop_detections(pipe):
    
    for item in pipe:

        outputs = []
        for src, img, oup, pred in zip(item['original_image'], item['image'], item['output'], item['pred']):
            sw, sh, _ = src.shape
            iw, ih, _ = img.shape
            scale_w, scale_h = sw/iw, sh/ih
        
            crops = [oup]
            for p in pred:
                x0, x1 = int(p[1] * scale_w), int(p[3] * scale_w)
                y0, y1 = int(p[0] * scale_h), int(p[2] * scale_h)
                
                # check for zero or negative crops
                if x1 <= x0 or y1 <= y0:
                    continue

                cropped = src[x0:x1, y0:y1]
                crops.append(cropped)
            
            outputs.append(crops)
        
        item['output'] = outputs
        
        yield item

