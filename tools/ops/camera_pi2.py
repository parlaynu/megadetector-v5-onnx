from itertools import count


def load_from_picamera2(params, width, height):
    from picamera2 import Picamera2, Preview
    import libcamera

    # parse the parameters
    cam_name = "picam2"
    hflip = vflip = False
    
    for param in params:
        param = param.lower()
        if param == "hflip":
            hflip = True
        elif param == "vflip":
            vflip = True
        else:
            cam_name = param
    
    # create and configure the camera
    picam2 = Picamera2()
    # print(picam2.sensor_modes)
    # print(picam2.camera_controls)
    # print(picam2.camera_properties)

    camera_config = picam2.create_still_configuration(
        transform=libcamera.Transform(hflip=hflip, vflip=vflip),
        main={
            'size': (width, height),
            'format': 'RGB888'
        }
    )
    picam2.align_configuration(camera_config)
    picam2.configure(camera_config)
    
    print("camera configuration:")
    print(camera_config)

    # start capturing
    picam2.start_preview(Preview.NULL)
    picam2.start()
    
    # capture a few frames to get the white balance right
    for idx in range(10):
        picam2.capture_array("main")
    
    # enter the main loop
    for idx in count():
        print(f"{idx:06d} reading image frame")

        img = picam2.capture_array("main")
    
        item = {
            'path': [f"{cam_name}_{idx:06d}.jpg"],
            'image': [img]
        }
        yield item

