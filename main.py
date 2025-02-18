import os
import re
import sys
import cv2
import time
import torch
import easyocr
import numpy as np
from ripikutils.stream import VideoStream

from jspl_hsm_coil_id_mapping.model import Reader, CoilTracker
from jspl_hsm_coil_id_mapping.heuristics import OCRBuffer

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )
    return result

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Reader(model_path="microsoft/Florence-2-large", device=device)
    coil_tracker = CoilTracker(model=None)
    ocr_buffer = OCRBuffer()
    cap = VideoStream("rtsp://admin:Ripik.ai@10.37.0.106:554")
    pattern = r'\b\d{10}\b'
    while True:
        # read a frame
        # print("Infe")
        a = time.time()
        frame = cap.read()
        if frame is None:
            continue
        # if not ret:
        #     break
        crop = rotate_image(frame[800:1700, 1500:2300], 50)
        cv2.imwrite("crop.jpg", crop)
        text = model.predict(crop)
        text = re.findall(pattern, text)
        # cv2.imwrite("frame.jpg", frame)
        # if text:
        #     cv2.imwrite(f"Predictions_17-02-2025/{text[0]}.png", frame)
        #     print(text)
        # print(text)
        # find if coil is present or not
        last_coil_status = coil_tracker.get_status()
        coil_tracker.add(frame, is_id_present=1 if text else 0)
        curr_coil_status = coil_tracker.get_status()
        # print(last_coil_status, curr_coil_status)
        if (not last_coil_status) and curr_coil_status:
            t_last_id_push = None
            t_track_start = time.time()
            ocr_buffer.empty()
        # If coil is present then start the ocr
        if curr_coil_status:
            if text:
                ocr_buffer.add(text[0])
            # Get an intermediate ID - Send to backend
            if (t_last_id_push == None) or (t_last_id_push - time.time()) > 30:
                id = ocr_buffer.get()
                cv2.imwrite(f"Pipeline-preds/{id}.png", frame)
                if not id:
                    continue
                # Send a response to backend
                pass
        else:
            # Send no coil present
            pass
        print(time.time() - a)
            
                