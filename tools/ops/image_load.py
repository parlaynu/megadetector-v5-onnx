import os
from itertools import count

import cv2


def load_images(image_src, recurse, extensions):

    if os.path.isfile(image_src):
        yield from _load_from_file(image_src, 0)
    
    elif os.path.isdir(image_src):
        yield from _load_from_dir(image_src, recurse, extensions)
    

def _load_from_dir(image_src, recurse, extensions):
    
    extensions = extensions.split(',')
    extensions = {"."+e.lower() for e in extensions}

    dirs = list()
    dirs.append(image_src)
    
    idx = 0
    while(len(dirs) > 0):
        dpath = dirs.pop()

        entries = os.listdir(dpath)
        entries.sort()
        
        for entry in entries:
            if entry.startswith("."):
                continue
            
            epath = os.path.join(dpath, entry)
            if recurse and os.path.isdir(epath):
                dirs.append(epath)
                continue
            
            if os.path.isfile(epath):
                _, ext = os.path.splitext(epath)
                if ext.lower() not in extensions:
                    continue
                
                idx += 1
                
                yield from _load_from_file(epath, idx)
        
        dirs.sort(reverse=True)


def _load_from_file(image_src, idx):
    
    print(f"{idx:04d} loading image {image_src}")

    # try and load it as a regular image
    img = cv2.imread(image_src)
    if img is not None:
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        item = {
            'path': [image_src],
            'image': [img]
        }
        yield item

    else:
        # try it like a video
        cap = cv2.VideoCapture(image_src)
        root, ext = os.path.splitext(image_src)
    
        for idx in count():
            ret, img = cap.read()
            if ret == False:
                break
        
            print(f"- loading video frame {idx:06d}")
        
            item = {
                'path': [f"{root}_{idx:06d}.jpg"],
                'image': [img]
            }
            yield item

        cap.release()

