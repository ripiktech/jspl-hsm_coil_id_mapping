import os
import re
import sys
import cv2
import time
import torch
import boto3
import easyocr
import logging
import numpy as np
from ripikutils.stream import VideoStream
from ripikutils.logsman import setup_logger
from ripikvisionpy.commons.clientMeta.ClientMetaWrapperV3 import ClientMetaWrapperV3

from jspl_hsm_coil_id_mapping import constants
from jspl_hsm_coil_id_mapping.model import Reader, CoilTracker, GroundingDinoPredictor
from jspl_hsm_coil_id_mapping.heuristics import OCRBuffer
from jspl_hsm_coil_id_mapping.utils import (
    rotate_image,
    get_response_with_s3_links,
    push_data_to_mongo,
    prepare_response,
    rotate_around_center
)

client_meta_wrapper = ClientMetaWrapperV3(version=1)
client_meta_stage_wrapper = ClientMetaWrapperV3(env='R_STAGE', version=1)

if __name__ == "__main__":
    CAMERA_ID = "camcoil"
    CLIENT_ID = "jindalsteel-stage"
    setup_logger(
        name=__name__,
        log_filename=f"{CAMERA_ID}_coil_id.log",
        max_log_size=50,
    )
    logging.info(f"INFO: Starting coil id tracking for {CLIENT_ID} and {CAMERA_ID}")
    if '-stage' in CLIENT_ID:
        client_meta_wrapper = client_meta_stage_wrapper
        constants.AWS_BUCKET_NAME = 'rpk-clnt-in-dev'
    client_meta = client_meta_wrapper.fetch_client_info(CLIENT_ID, constants.USE_CASE, True)
    if client_meta is None:
        logging.info("Unable to fetch client meta.")
        raise Exception("Unable to fetch client meta.")
    else:
        logging.info(f"INFO: Fetched client meta information for {CLIENT_ID}")
    s3 = boto3.client('s3')
    tracking_model = GroundingDinoPredictor(
        grounding_text=constants.COIL_DET_GROUNDING_TEXT, 
        confidence_threshold=constants.COIL_DET_CONF, 
        model_config_path=constants.COIL_DET_MODEL_PATH,
        device=constants.DEVICE
    )
    model = Reader(model_path=constants.MODEL_PATH, device=constants.DEVICE)
    coil_tracker = CoilTracker(model=tracking_model)
    ocr_buffer = OCRBuffer()
    cap = VideoStream(constants.CONFIG[CAMERA_ID]["rtsp"])
    crop_coords = constants.CONFIG[CAMERA_ID]["crop_coords"]
    rotation_angle = constants.CONFIG[CAMERA_ID]["rotation_angle"]
    pattern = r'\b\d{10}\b'
    ocr_img = None
    ocr_img_crop = None
    t_last_id_push = None
    entity_id = None
    while True:
        try:
            frame = cap.read()
            if frame is None:
                logging.warning("Receving invalid frames.")
                time.sleep(1.0)
                continue
            coil_crop = frame[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]] 
            crop = rotate_image(coil_crop, rotation_angle)
            # <TEMPORARY CODE>
            cv2.imwrite("crop.jpg", coil_crop)
            # </TEMPORARY CODE>
            text = model.predict(crop)
            text = re.findall(pattern, text)
            last_coil_status = coil_tracker.get_status()
            coil_tracker.add(coil_crop)
            curr_coil_status = coil_tracker.get_status()

            # New coil entered -> initalize buffers.
            if not(last_coil_status) and curr_coil_status:
                ocr_buffer.empty()
                t_last_id_push = None
                ocr_img = None
                ocr_img_crop = None
                entity_id = int(time.time() * 1000)
                logging.info("New coil appeared.")
            push_to_backend = False
            is_final_push = False
            # If coil present and has a valid OCR.
            if curr_coil_status and len(text) > 0:
                ocr_buffer.add(text[0])
                ocr_img = frame.copy()
                ocr_img_crop = coil_crop.copy()
                # Push coilId to backend in 30s intervals.
                # Send a response to backend
                if (t_last_id_push == None) or (t_last_id_push - time.time()) > constants.ID_PUSH_COOLDOWN:
                    push_to_backend = True
                    
            elif last_coil_status and \
                 not(curr_coil_status):
                # Final id push for a coil
                push_to_backend = True
                is_final_push = True

            push_to_backend = push_to_backend or (t_last_id_push == None)
            push_to_backend = (
                push_to_backend or 
                (time.time() - t_last_id_push) > 30
            )
            if not(last_coil_status) and not(curr_coil_status):
                ocr_buffer.empty()
            if push_to_backend:
                # if ocr_img
                ocr_id = ocr_buffer.get()
                if ocr_img is not None:
                    original_image = ocr_img.copy()
                    annotated_image = ocr_img_crop.copy()
                else:
                    original_image = frame.copy()
                    annotated_image = coil_crop.copy()
                model_response = prepare_response(
                    client_id=CLIENT_ID,
                    camera_id=CAMERA_ID,
                    ocr_id=ocr_id,
                    entity_id=entity_id,
                    is_coil_present=(last_coil_status == 1),
                    is_alert=(is_final_push and ocr_id == ""),
                    original_image=rotate_around_center(original_image, 7.0),
                    annotated_image=annotated_image
                )
                backend_response = get_response_with_s3_links(s3, model_response, aws_bucket_name=constants.AWS_BUCKET_NAME)
                push_data_to_mongo(
                    response=backend_response,
                    client_meta_wrapper=client_meta_wrapper,
                    client_meta=client_meta,
                )
                t_last_id_push = time.time()
                if is_final_push:
                    # Empty out buffers and images
                    logging.info(f"Coil at Time: {entity_id} with ID: {ocr_id} exited.")
                    cv2.imwrite(f"Pipeline-preds/{entity_id}_{ocr_id}.png", ocr_img)
                    ocr_img = None
                    ocr_img_crop = None
                    entity_id = None
                    ocr_buffer.empty()
                    # wait for the robot gate to close.
                    time.sleep(5.0)
        except Exception as e:
            logging.warning(str(e))
            print(str(e))