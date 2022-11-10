from itertools import count
import time


def load_from_picamera2(cam_name, width, height):
    from picamera2 import Picamera2, Preview
    
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start_preview(Preview.NULL)
    picam2.start()
    
    for idx in count():
        print(f"{idx:06d} reading image frame")

        img = picam2.capture_array("main")
        img = img[..., :3]
    
        item = {
            'path': [f"{cam_name}_{idx:06d}.jpg"],
            'image': [img]
        }
        yield item

