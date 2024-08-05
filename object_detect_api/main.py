import cv2
import base64
import numpy as np
from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
from ultralytics import YOLO
#from collections import defaultdict
#from ultralytics.utils.plotting import Annotator, colors

#model = YOLO("./yolov8/yolov8n-pose.pt", task = "pose")
model = YOLO("./yolov8/yolov8n-seg.pt", task = "segment")
names = model.model.names
#track_history = defaultdict(lambda: [])

print('===> OpenCV Version.....:', cv2.__version__, flush = True)
print('===> OpenCV Cuda devices:', cv2.cuda.getCudaEnabledDeviceCount(), flush = True)
#print('===> Opencv build information:', cv2.getBuildInformation(), flush = True)

app = FastAPI()

class Params(BaseModel):
    rgb_image: Union[str, None]

@app.post("/draw_object/")
def detect_segment(params: Params = None):

    try:
        if not params is None:

            segment_frame = base64.b64decode(params.rgb_image)
            segment_frame = np.frombuffer(segment_frame, np.uint8)
            segment_frame = cv2.imdecode(segment_frame, cv2.IMREAD_COLOR)
            results = model.track(segment_frame, persist=True)

            object_id = None
            if not results[0].boxes.id is None:
                object_id = results[0].boxes.id.int().cpu().tolist()

                object_mask = None
                if not results[0].masks.xy is None:
                    object_mask = [mask.tolist() for mask in results[0].masks.xy]

                    object_class = [results[0].names[obj_class] for obj_class in results[0].boxes.cls.int().cpu().tolist()]
                    object_conf = results[0].boxes.conf.float().cpu().tolist()

                    object_list = list(zip(object_id, object_class, object_conf, object_mask))
                    object_list.sort(key=lambda x: x[1])

        return {'object_list': object_list}
    
    except Exception as e:
        print(e, flush=True)

    return None

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)