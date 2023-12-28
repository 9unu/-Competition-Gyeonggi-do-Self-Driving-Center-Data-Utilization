from detection_helpers import *
from tracking_helpers import *
from bridge_wrapper import *
from PIL import Image
import gc
import time


# output = None will not save the output video
num = 90
cctv = 3
road = 2
start = 1
for i in range(start, num + 1):
    try: # begin video capture
        detector = Detector(classes = [2,3,5,7]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
        detector.load_model('./yolov7x.pt',) # pass the path to the trained weight file

        # Initialise  class that binds detector and tracker in one class
        tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

        tracker.track_video("./IO_data/input/video/cctv"+str(cctv)+"-"+str(road)+"/CCTV_" + str(cctv)+ "_"+str(i)+".mp4",
         output="./IO_data/output/video/cctv"+str(cctv)+"-"+str(road)+"/CCTV_" + str(cctv)+ "_"+str(i)+".mp4",
         csv_file="./IO_data/output/csv/cctv"+str(cctv)+"-"+str(road)+"/CCTV_" + str(cctv)+ "_"+str(i)+".csv",
         show_live = False, skip_frames = 2, count_objects = True, verbose=1) 
        gc.collect()
    except:
        pass