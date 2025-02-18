import time
import easyocr
from collections import deque
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoProcessor



class CoilTracker:
    def __init__(self, model=None, buffer_length=100, window_size=50):
        self.model = model
        self.buffer_length = buffer_length
        self.window_size = window_size
        self.tracking_buffer = []

    def _add_to_buffer(self, is_id_present):
        if len(self.tracking_buffer) < self.buffer_length:
            self.tracking_buffer.append(is_id_present)
        else:
            self.tracking_buffer.pop(0)
            self.tracking_buffer.append(is_id_present)

    def _predict(self, frame):
        is_id_present = self.model(frame)
        return is_id_present

    def add(self, frame, is_id_present=0):
        if self.model:
            is_id_present = self._predict(frame)
        self._add_to_buffer(is_id_present)
    
    def get_status(self):
        if sum(self.tracking_buffer[-self.window_size:]) == 0:
            return 0
        return 1

class WeightedCoilTracker:
    def __init__(self, model, buffer_length=100, time_window=5.0):
        """
        Initialize the weighted coil tracker
        
        Args:
            model: Detection model
            buffer_length (int): Maximum number of entries in buffer
            time_window (float): Time window in seconds to consider for recent predictions
        """
        self.model = model
        self.buffer_length = buffer_length
        self.time_window = time_window
        # Store (timestamp, prediction) pairs
        self.tracking_buffer = deque(maxlen=buffer_length)
        
    def _add_to_buffer(self, prediction_data: Tuple[float, bool]):
        """Add timestamp and prediction to buffer"""
        self.tracking_buffer.append(prediction_data)
            
    def _predict(self, frame) -> bool:
        """Run model prediction"""
        is_coil_present = self.model(frame)
        return is_coil_present
    
    def _get_weighted_predictions(self, window_size: int = 10) -> List[float]:
        """
        Get weighted predictions considering timestamps
        
        Args:
            window_size: Number of recent predictions to consider
            
        Returns:
            List of weighted predictions (0 to 1)
        """
        if not self.tracking_buffer:
            return []
            
        recent_preds = list(self.tracking_buffer)[-window_size:]
        if not recent_preds:
            return []
            
        weighted_preds = []
        current_time = time.time()
        for i in range(len(recent_preds)):
            timestamp, pred = recent_preds[i]
            # Calculate time difference
            time_diff = current_time - timestamp
            # Apply exponential decay weight based on time difference
            weight = max(0, 1 - (time_diff / self.time_window))
            # If frames are too old, give them very low weight
            if time_diff > self.time_window:
                weight = 0.1
            # Calculate weighted prediction
            weighted_pred = pred * weight
            weighted_preds.append(weighted_pred)
        return weighted_preds
    
    def add(self, frame):
        """Process a new frame"""
        current_time = time.time()
        is_coil_present = self._predict(frame)
        self._add_to_buffer((current_time, is_coil_present))
    
    def get_status(self) -> int:
        """
        Get current coil status considering weighted predictions
        
        Returns:
            0 if no coil detected, 1 if coil detected
        """
        weighted_preds = self._get_weighted_predictions(window_size=10)
        if not weighted_preds:
            return 0
            
        # Calculate weighted average
        weighted_avg = sum(weighted_preds) / len(weighted_preds)
        # Use threshold to determine presence
        threshold = 0.3  # Can be adjusted based on requirements
        return 1 if weighted_avg > threshold else 0
        
    def get_buffer_health(self) -> float:
        """
        Get a metric indicating buffer health based on frame timing
        
        Returns:
            Float between 0 and 1 indicating buffer health
        """
        if len(self.tracking_buffer) < 2:
            return 1.0
            
        timestamps = [t for t, _ in self.tracking_buffer]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 1.0
            
        # Calculate average frame interval
        avg_interval = sum(intervals) / len(intervals)
        
        # Calculate variance in intervals
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        
        # High variance indicates irregular frame timing
        health_score = 1.0 / (1.0 + variance)
        return health_score

class Reader:
    def __init__(self, model_path, device = "cpu"):
        self.prompt = "<OCR>"
        self.device = device
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def predict(self, image):   
        inputs = self.processor(
            text=self.prompt, 
            images=image, 
            return_tensors="pt").to(self.device)
        with torch.autocast(device_type="cuda"):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        return generated_text