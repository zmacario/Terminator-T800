import cv2
import numpy as np
from typing import Mapping, Tuple

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

_THICKNESS_TESSELATION = 1
_THICKNESS_CONTOURS = 1
_FACEMESH_CONTOURS_CONNECTION_STYLE = {
    face_mesh_connections.FACEMESH_LIPS:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYE:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYEBROW:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYE:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_FACE_OVAL:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS)
}

def get_default_face_mesh_contours_style() -> Mapping[Tuple[int, int], DrawingSpec]:

    default_style = _FACEMESH_CONTOURS_CONNECTION_STYLE
    face_mesh_contours_connection_style = {}
    for k, v in default_style.items():
        for connection in k:
            face_mesh_contours_connection_style[connection] = v
    return face_mesh_contours_connection_style

def get_default_face_mesh_tesselation_style() -> DrawingSpec:

    return DrawingSpec(color=_WHITE, thickness=_THICKNESS_TESSELATION)

def get_default_face_mesh_iris_connections_style() -> Mapping[Tuple[int, int], DrawingSpec]:

    face_mesh_iris_connections_style = {}
    left_spec = DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS)
    for connection in face_mesh_connections.FACEMESH_LEFT_IRIS:
        face_mesh_iris_connections_style[connection] = left_spec

    right_spec = DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS)
    for connection in face_mesh_connections.FACEMESH_RIGHT_IRIS:
        face_mesh_iris_connections_style[connection] = right_spec

    return face_mesh_iris_connections_style

def draw_landmarks(rgb_frame, face_landmarks_list):

    if not face_landmarks_list is None:
        if len(face_landmarks_list) > 0:

            face_landmarks = face_landmarks_list[0]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark['x'], y=landmark['y'], z=landmark['z']) for landmark in face_landmarks])

            face_frame = np.copy(rgb_frame)
            H, W = rgb_frame.shape[0], rgb_frame.shape[1]
            R = H / W

            solutions.drawing_utils.draw_landmarks(image=face_frame,
                                                landmark_list=face_landmarks_proto,
                                                connections=solutions.face_mesh.FACEMESH_TESSELATION,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=get_default_face_mesh_tesselation_style())
                
            solutions.drawing_utils.draw_landmarks(image=face_frame,
                                                landmark_list=face_landmarks_proto,
                                                connections=solutions.face_mesh.FACEMESH_CONTOURS,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=get_default_face_mesh_contours_style())
                
            solutions.drawing_utils.draw_landmarks(image=face_frame,
                                                landmark_list=face_landmarks_proto,
                                                connections=solutions.face_mesh.FACEMESH_IRISES,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=get_default_face_mesh_iris_connections_style())
            
            non_zero_indices = np.nonzero(face_frame[:, :, 0])
            min_y = np.min(non_zero_indices[0])
            min_x = np.min(non_zero_indices[1])
            max_y = np.max(non_zero_indices[0])
            max_x = np.max(non_zero_indices[1])
            face_frame = face_frame[min_y: max_y, min_x: max_x]

            h = face_frame.shape[0]
            w = face_frame.shape[1]
            delta_h = H - h
            delta_w = W - w

            if delta_h > delta_w:
                delta_h = delta_w * R
                face_frame = cv2.resize(face_frame, (W, int(h + delta_h)), interpolation=cv2.INTER_LINEAR)

                h = face_frame.shape[0]
                delta_h = int((H - h) / 2)
                aux_frame = np.zeros((H, W, 3), dtype = np.uint8)
                aux_frame[delta_h: h + delta_h, :, :] = face_frame
                face_frame = aux_frame

            else:
                delta_w = delta_h / R
                face_frame = cv2.resize(face_frame, (int(w + delta_w), H), interpolation=cv2.INTER_LINEAR)

                w = face_frame.shape[1]
                delta_w = int((W - w) / 2)
                aux_frame = np.zeros((H, W, 3), dtype = np.uint8)
                aux_frame[:, delta_w: w + delta_w, :] = face_frame
                face_frame = aux_frame

            return face_frame
        
    return None