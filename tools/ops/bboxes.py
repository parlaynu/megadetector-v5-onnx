import numpy as np
import cv2


def draw_bboxes(pipe):

    for item in pipe:
        item['output'] = []
        for src, offset, img, pred in zip(item['original_image'], item['offset'], item['image'], item['pred']):
            sh, sw, _ = src.shape
            ih, iw, _ = img.shape
            oh, ow = offset
            
            scale_w, scale_h = sw/(iw - 2*ow), sh/(ih - 2*oh)
        
            oup = np.copy(src)
            for p in pred:
                category = p[5]
                if category == 0:      # animal
                    color = (0,0,255)
                elif category == 1:    # human (??)
                    color = (255,0,0)
                else:
                    color = (0,255,255)
                
                x1, y1, x2, y2 = p[:4]
                
                tl = (int((x1 - ow) * scale_w), int((y1 - oh) * scale_h))
                br = (int((x2 - ow) * scale_w), int((y2 - oh) * scale_h))
                oup = cv2.rectangle(oup, tl, br, color, 2)
            
            item['output'].append(oup)
        
        yield item
