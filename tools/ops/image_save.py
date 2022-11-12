import os
import cv2


def save_images(pipe, inp_root, out_root, save_all):

    os.makedirs(out_root, exist_ok=True)

    for item in pipe:

        for ipath, output, pred in zip(item['path'], item['output'], item['pred']):
            if save_all == False and len(pred) == 0:
                continue

            irelative = ipath if inp_root is None else ipath[len(inp_root):]
            if irelative.startswith('/'):
                irelative = irelative[1:]
            opath = os.path.join(out_root, irelative)
            odir = os.path.dirname(opath)
            os.makedirs(odir, exist_ok=True)

            root, ext = os.path.splitext(opath)
            if isinstance(output, (list, set)):
                for idx, oup in enumerate(output):
                    opath = f"{root}_{idx:04d}{ext}"
                    cv2.imwrite(opath, oup)
            else:
                opath = f"{root}_0000{ext}"
                cv2.imwrite(opath, output)

        yield item

