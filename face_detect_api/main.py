import cv2
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union#, Mapping, Tuple

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'), running_mode=VisionRunningMode.VIDEO)

class Params(BaseModel):
    rgb_image: Union[str, None]
    frame_timestamp_ms: Union[int, None] = 1

app = FastAPI()
@app.post("/draw_face/")
def draw_face(params: Params = None):
    
    try:
        if not params is None:

            mp_image = base64.b64decode(params.rgb_image)
            mp_image = np.frombuffer(mp_image, np.uint8)
            mp_image = cv2.imdecode(mp_image, cv2.IMREAD_COLOR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)
            with FaceLandmarker.create_from_options(options) as landmarker:
                    
                    face_landmarker_result = landmarker.detect_for_video(mp_image, params.frame_timestamp_ms)  
                    return {'face_landmarks': face_landmarker_result.face_landmarks}  
    
    except Exception as e:
        print(e, flush=True)

    return None

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)