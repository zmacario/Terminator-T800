import cv2
import torch
import base64
import numpy as np

from fastapi import FastAPI
from typing import Union #, List
from pydantic import BaseModel
from depth_anything_v2.dpt import DepthAnythingV2

print('===> Torch Version:', torch.__version__, flush = True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('===> Torch Cuda Devices:', DEVICE, flush = True)

model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                 'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}

encoder = 'vits' # or 'vits', 'vitb', 'vitg', 'vitl'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

app = FastAPI()

class Params(BaseModel):
    rgb_image: Union[str, None]

@app.post("/draw_depth/")
def detect_depth(params: Params = None):
    
    try:
        if not params is None:

            rgb_frame = base64.b64decode(params.rgb_image)
            rgb_frame = np.frombuffer(rgb_frame, np.uint8)
            rgb_frame = cv2.imdecode(rgb_frame, cv2.IMREAD_COLOR)
            
            depth_frame = model.infer_image(rgb_frame)
            depth_frame = (depth_frame - depth_frame.min()) / (depth_frame.max() - depth_frame.min()) * 255.0
            depth_frame = depth_frame.astype(np.uint8)
            depth_frame = np.repeat(depth_frame[..., np.newaxis], 3, axis=-1)
            _, depth_frame = cv2.imencode('.jpg', depth_frame)
            depth_frame = base64.b64encode(depth_frame).decode('utf-8')

        return {'result_frame': depth_frame}
    
    except Exception as e:
        print(e, flush=True)

    return None

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)