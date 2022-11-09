import os
import cv2


def find_images(image_dir, recurse, extensions):

    extensions = extensions.split(',')
    extensions = {"."+e.lower() for e in extensions}

    dirs = list()
    dirs.append(image_dir)
    
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
                
                print(f"{idx:04d} found image {epath}")
                idx += 1
                
                item = {
                    'path': [epath]
                }
                yield item
        
        dirs.sort(reverse=True)


def load_image(pipe, width, height, nchans, preserve_aspect):
    
    for item in pipe:
        img = cv2.imread(item['path'][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        item['original_image'] = [img]

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

        item['image'] = [img]

        yield item

