import cv2
import base64
import numpy as np
import mediapipe as mp

from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'), running_mode=VisionRunningMode.VIDEO)

def draw_landmarks_on_frame(rgb_frame, detection_result):

    face_frame = None
    face_landmarks_list = detection_result.face_landmarks
    if not face_landmarks_list is None:
        if len(face_landmarks_list) > 0:

            face_frame = np.copy(rgb_frame)
            idx = 0
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

            solutions.drawing_utils.draw_landmarks(image=face_frame,
                                                        landmark_list=face_landmarks_proto,
                                                        connections=solutions.face_mesh.FACEMESH_TESSELATION,
                                                        landmark_drawing_spec=None,
                                                        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                
            solutions.drawing_utils.draw_landmarks(image=face_frame,
                                                        landmark_list=face_landmarks_proto,
                                                        connections=solutions.face_mesh.FACEMESH_CONTOURS,
                                                        landmark_drawing_spec=None,
                                                        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style())
                
            solutions.drawing_utils.draw_landmarks(image=face_frame,
                                                        landmark_list=face_landmarks_proto,
                                                        connections=solutions.face_mesh.FACEMESH_IRISES,
                                                        landmark_drawing_spec=None,
                                                        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    
    return face_frame

app = FastAPI()

class Params(BaseModel):
    rgb_image: Union[str, None]
    frame_timestamp_ms: Union[int, None] = 1

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
                    if not face_landmarker_result is None:
                        
                        face_frame = draw_landmarks_on_frame(mp_image.numpy_view(), face_landmarker_result)
                        ret, face_frame = cv2.imencode('.jpg', face_frame)
                        if ret is True:

                            face_frame = base64.b64encode(face_frame).decode('utf-8')
                            return {'result_frame': face_frame}
    
    except Exception as e:
        print(e, flush=True)

    return None

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)