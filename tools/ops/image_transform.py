import numpy as np
import cv2


def transform_images(pipe, width, height, nchans, preserve_aspect):
    
    for item in pipe:
        
        originals = []
        images = []
        inputs = []
        
        imgs = item['image']
        for img in imgs:
            
            originals.append(img)
            
            img = _resize_pad(img, width, height, nchans, preserve_aspect)
            images.append(img)

            inp = img.astype(np.float32) / 255.0
            inp = np.transpose(inp, axes=[2, 0, 1])
            inp = inp[np.newaxis,...]
            inp = np.ascontiguousarray(inp)
        
            inputs.append(inp)

        item['original_image'] = originals
        item['image'] = images
        item['input'] = np.concatenate(inputs, axis=0)
        
        yield item

        

def _resize_pad(img, width, height, nchans, preserve_aspect):
    iheight, iwidth, _ = img.shape
    if iheight != height or iwidth != width:
        nwidth, nheight = width, height
        if preserve_aspect:
            scale = min(width/iwidth, height/iheight)
            nwidth, nheight = int(scale*iwidth), int(scale*iheight)
        
        print(f"- resizing from {iwidth}x{iheight} to {nwidth}x{nheight}")
        img = cv2.resize(img, (nwidth, nheight), interpolation=cv2.INTER_LINEAR)
        
        if nwidth != width or nheight != height:
            top = int((height - nheight)/2)
            bottom = height - nheight - top
            left = int((width - nwidth)/2)
            right = width - nwidth - left

            print(f"- padding from {nwidth}x{nheight} to {width}x{height}")
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return img

