from itertools import count
import cv2


def load_from_jetson_csi(params, width, height):

    # parse the parameters
    cam_name = "nano_csi"
    hflip = vflip = False
    
    for param in params:
        param = param.lower()
        if param == "hflip":
            hflip = True
        elif param == "vflip":
            vflip = True
        else:
            cam_name = param
    
    device = 0
    cap_width = 1920
    cap_height = 1080
    framerate = 30
    
    cam_str = " ".join([f"nvarguscamerasrc sensor-id={device} !",
                f"video/x-raw(memory:NVMM),width={cap_width},height={cap_height},format=(string)NV12,framerate=(fraction){framerate}/1 !",
                f"nvvidconv !",
                f"video/x-raw,width=(int){width},height=(int){height},format=(string)BGRx !",
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
            break
        
        item = {
            'path': [f"{cam_name}_{idx:06d}.jpg"],
            'image': [img]
        }
        yield item

