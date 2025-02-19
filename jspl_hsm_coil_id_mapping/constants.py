import torch
MODEL_PATH = "microsoft/Florence-2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID_PUSH_COOLDOWN = 30
RESIZE_IMAGE_PERCENTAGE = 100
AWS_BUCKET_NAME = "rpk-clnt-in-dev"
CAMERA_GP_ID = "cameraGp1"
PLANT_ID = "angul"
USE_CASE = "coilidtracking"

## COIL DETECTOR PARAMS
COIL_DET_GROUNDING_TEXT = ".coil"
COIL_DET_CONF = 0.4
COIL_DET_MODEL_PATH = "IDEA-Research/grounding-dino-base"
CONFIG = {
    "camcoil": {
        "rtsp": "rtsp://admin:Ripik.ai@10.37.0.106:554",
        "crop_coords" : [800, 1700, 1500, 2300],
        "rotation_angle": 50
    }
}