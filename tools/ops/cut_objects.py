

def cut_objects(pipe):
    
    for item in pipe:

        outputs = []
        for src, offset, img, oup, pred in zip(item['original_image'], item['offset'], item['image'], item['output'], item['pred']):
            sh, sw, _ = src.shape
            ih, iw, _ = img.shape
            oh, ow = offset

            scale_w, scale_h = sw/(iw - 2*ow), sh/(ih - 2*oh)
        
            crops = [oup]
            for p in pred:
                x1, y1, x2, y2 = p[:4]

                x1, y1 = int((x1 - ow) * scale_w), int((y1 - oh) * scale_h)
                x2, y2 = int((x2 - ow) * scale_w), int((y2 - oh) * scale_h)
                
                # check for zero crops... can happen due to rounding if the
                #   input image is smaller than the processing resolution
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped = src[y1:y2, x1:x2]
                crops.append(cropped)
            
            outputs.append(crops)
        
        item['output'] = outputs
        
        yield item

