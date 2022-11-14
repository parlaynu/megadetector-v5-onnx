import os
from itertools import count

import cv2


def load_images(image_src, params):

    recurse = False
    if image_src.endswith("/..."):
        recurse = True
        image_src = image_src[:-4]
    
    if os.path.isfile(image_src):
        start_idx = 0 if len(params) == 0 else int(params[0])
        yield from _load_from_file(image_src, 0, start_idx)
    
    elif os.path.isdir(image_src):
        yield from _load_from_dir(image_src, params, recurse)
    

def _load_from_dir(image_src, image_exs, recurse):
    
    # format the extensions ready for checking
    image_exs = {e.lstrip(".").lower() for e in image_exs}  # get all extensions to known state
    image_exs = {"." + e for e in image_exs} # add back the leading '.'

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
                if ext.lower() not in image_exs:
                    continue
                
                yield from _load_from_file(epath, idx)

                idx += 1
        
        dirs.sort(reverse=True)


def _load_from_file(image_src, idx, start_idx=0):
    
    print(f"{idx:06d} loading {image_src}")

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
        
        # seek forward to starting frame in the video
        if start_idx > 0:
            print(f"- skipping {start_idx} video frames")
            for idx in range(start_idx):
                ret, img = cap.read()
                if ret == False:
                    break
    
        for idx in count(start_idx):
            ret, img = cap.read()
            if ret == False:
                break
        
            print(f"{idx:06d} reading video frame")
        
            item = {
                'path': [f"{root}_{idx:06d}.jpg"],
                'image': [img]
            }
            yield item

        cap.release()

