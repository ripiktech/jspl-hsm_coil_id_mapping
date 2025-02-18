import torch
MODEL_PATH = "microsoft/Florence-2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID_PUSH_COOLDOWN = 30
RESIZE_IMAGE_PERCENTAGE = 100
AWS_BUCKET_NAME = "rpk-clnt-in-dev"
CONFIG = {
    "camcoil": {
        "rtsp": "",
        "crop_coords" : [800, 1700, 1500, 2300],
        "rotation_angle": 50
    }
}