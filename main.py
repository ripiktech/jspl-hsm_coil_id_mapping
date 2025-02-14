import os
import sys
import easyocr
import time

from jspl_hsm_coil_id_mapping.model import Reader, CoilTracker
from jspl_hsm_coil_id_mapping.heuristics import OCRBuffer

if __name__ == "__main__":
    model = Reader()
    coil_tracker = CoilTracker(model=None)
    ocr_buffer = OCRBuffer()
    while True:
        # read a frame
        # find if coil is present or not
        last_coil_status = coil_tracker.get_status()
        coil_tracker.add(frame)
        curr_coil_status = coil_tracker.get_status()
        if (not last_coil_status) and curr_coil_status:
            t_last_id_push = None
            t_track_start = time.time()
            ocr_buffer.empty()
        # If coil is present then start the ocr
        if curr_coil_status:
            text = model.predict(frame)
            ocr_buffer.add(text)
            # Get an intermediate ID - Send to backend
            if (t_last_id_push == None) or (t_last_id_push - time.time()) > 30:
                id, occ = ocr_buffer.get()
                # Send a response to backend
                pass
        else:
            # Send no coil present
            pass
            
                