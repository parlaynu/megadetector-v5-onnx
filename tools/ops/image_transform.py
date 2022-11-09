import numpy as np


def transform_image(pipe):
    
    for item in pipe:
        imgs = item['image']
        
        inputs = []
        for img in imgs:
            inp = img.astype(np.float32) / 255.0
            inp = np.transpose(inp, axes=[2, 0, 1])
            inp = inp[np.newaxis,...]
            inp = np.ascontiguousarray(inp)
        
            inputs.append(inp)
        
        item['input'] = np.concatenate(inputs, axis=0)
        # print(f"- array shape {item['input'].shape}")
        
        yield item

        
