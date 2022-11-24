from itertools import count
import cv2


def load_from_jetson_csi(params, width, height):

    # default settings
    cam_name = "nano_csi"
    hflip = vflip = False
    device = 0
    cap_width = 1920
    cap_height = 1080
    framerate = 30

    # parse the parameters
    for param in params:
        param = param.lower()
        if param == "hflip":
            hflip = True
        elif param == "vflip":
            vflip = True
        elif param == "0" or param == "1":
            device = int(param)
        else:
            cam_name = param
    
    cam_str = " ".join([f"nvarguscamerasrc sensor-id={device} !",
                f"video/x-raw(memory:NVMM),width={cap_width},height={cap_height},format=(string)NV12,framerate=(fraction){framerate}/1 !",
                f"nvvidconv !",
                f"video/x-raw,width=(int){cap_width},height=(int){cap_height},format=(string)BGRx !",
                f"videoconvert !",
                f"appsink"
                ])
                
    cam = cv2.VideoCapture(cam_str, cv2.CAP_GSTREAMER)
    
    for idx in range(30):
        cam.read()
    
    for idx in count():
        print(f"{idx:06d} reading image frame")

        ok, img = cam.read()
        if ok == False:
            print("- read failed")
            break
        
        # REVISIT: find a gstreamer way to do this
        if hflip and vflip:
            img = cv2.flip(img, -1)
        elif hflip:
            img = cv2.flip(img, 1)
        elif vflip:
            img = cv2.flip(img, 0)
        
        item = {
            'path': [f"{cam_name}_{idx:06d}.jpg"],
            'image': [img]
        }
        yield item

