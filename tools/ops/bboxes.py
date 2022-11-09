import numpy as np
import cv2


def draw_bboxes(pipe):

    for item in pipe:
        item['output'] = []
        for src, img, pred in zip(item['original_image'], item['image'], item['pred']):
            sw, sh, _ = src.shape
            iw, ih, _ = img.shape
            scale_w, scale_h = sw/iw, sh/ih
        
            oup = np.copy(src)
            for p in pred:
                category = p[5]
                if category == 0:      # animal
                    color = (0,0,255)
                elif category == 1:    # human (??)
                    color = (255,0,0)
                else:
                    color = (0,255,255)
                    
                tl = (int(p[0] * scale_h), int(p[1] * scale_w))
                br = (int(p[2] * scale_h), int(p[3] * scale_w))
                oup = cv2.rectangle(oup, tl, br, color, 2)
            
            item['output'].append(oup)
        
        yield item
