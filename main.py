import os
import re
import sys
import cv2
import time
import torch
import boto3
import easyocr
import numpy as np
from ripikutils.stream import VideoStream

from jspl_hsm_coil_id_mapping import constants
from jspl_hsm_coil_id_mapping.model import Reader, CoilTracker
from jspl_hsm_coil_id_mapping.heuristics import OCRBuffer
from jspl_hsm_coil_id_mapping.utils import (
    rotate_image,
    get_response_with_s3_links,
    push_data_to_mongo
)

if __name__ == "__main__":
    PLANT_ID = "angul"
    CAMERA_ID = "camcoil"
    s3 = boto3.client('s3')
    model = Reader(model_path=constants.MODEL_PATH, device=constants.DEVICE)
    coil_tracker = CoilTracker(model=None)
    ocr_buffer = OCRBuffer()
    cap = VideoStream(constants.CONFIG[CAMERA_ID]["rtsp"])
    crop_coords = constants.CONFIG[CAMERA_ID]["crop_coords"]
    rotation_angle = constants.CONFIG[CAMERA_ID]["rotation_angle"]
    pattern = r'\b\d{10}\b'
    ocr_img = None
    entity_id = None
    while True:
        # read a frame
        a = time.time()
        frame = cap.read()
        if frame is None:
            continue
        crop = rotate_image(frame[
            crop_coords[0]:crop_coords[1],
            crop_coords[2]:crop_coords[3]], 
            rotation_angle
        )

        cv2.imwrite("crop.jpg", crop)
        text = model.predict(crop)
        text = re.findall(pattern, text)
        last_coil_status = coil_tracker.get_status()
        coil_tracker.add(frame, is_id_present=1 if text else 0)
        curr_coil_status = coil_tracker.get_status()

        # Initialize a new coil buffer.
        if (not last_coil_status) and curr_coil_status:
            entity_id = time.time()
            ocr_img = frame.copy()
            t_last_id_push = None
            t_track_start = time.time()
            ocr_buffer.empty()

        # If coil is present then start the ocr
        if curr_coil_status:
            if text:
                ocr_buffer.add(text[0])
            # Get an intermediate ID - Send to backend
            if (t_last_id_push == None) or (t_last_id_push - time.time()) > constants.ID_PUSH_COOLDOWN:
                ocr_id = ocr_buffer.get()
                cv2.imwrite(f"Pipeline-preds/{entity_id}_{ocr_id}.png", ocr_img)
                if not id:
                    continue
                model_response = dict()
                model_response["coilId"] = ocr_id
                model_response["duplicateIds"] = []
                model_response["isCoilPresent"] = curr_coil_status
                model_response["isAlert"] = False
                model_response["originalImage"] = frame.copy()
                model_response["annotatedImage"] = frame.copy()
                model_response["cameraGrpId"] = "cameraGp1"
                model_response["plantId"] = PLANT_ID
                model_response["cameraId"] = CAMERA_ID
                # backend_response = get_response_with_s3_links(
                #     s3,
                #     model_response,
                #     aws_bucket_name=constants.AWS_BUCKET_NAME
                # )
                # push_data_to_mongo()
                # Send a response to backend
        else:
            # Send no coil present
            pass
        print(time.time() - a)
            
                