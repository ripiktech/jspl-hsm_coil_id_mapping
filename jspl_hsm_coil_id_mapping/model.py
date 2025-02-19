import time
import easyocr
from collections import deque
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoProcessor, GroundingDinoForObjectDetection
import PIL
from numpy import ndarray
import cv2
from PIL import Image, ImageDraw, ImageFont

def draw_bboxes_with_labels(image, bbox_dict): 
    '''
    Annotate an image with bounding boxes and labels.
    Args:
        image (PIL.Image): The input image
        bbox_dict (dict): A dictionary mapping labels to lists of bounding boxes
    Returns:
        PIL.Image: The annotated image
    '''
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    color = 'red'
    # Annotate each bounding box
    for label, bboxes in bbox_dict.items():
        for bbox in bboxes:
            # Unpack coordinates
            x1, y1, x2, y2 = bbox
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Add label text
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text above the bounding box
            text_x = x1
            text_y = max(0, y1 - text_height - 5)
            
            # Draw text background
            draw.rectangle([text_x, text_y, text_x + text_width + 4, text_y + text_height + 4], 
                           fill=color, outline=color)
            # Draw text
            draw.text((text_x + 2, text_y + 2), label, font=font, fill=(255, 255, 255))
    return image


class GroundingDinoPredictor:
    """
    A predictor class for object detection using the Grounding DINO model.

    This class provides an interface to perform zero-shot object detection 
    using a pre-trained Grounding DINO model, which can detect objects 
    based on text descriptions.

    Attributes:
        grounding_text (str): Text description of objects to detect.
        confidence_threshold (float): Minimum confidence score for object detection.
        config_path (str): Path to the model configuration.
        device (str): Computing device to run the model on (default is "cpu").
        model (GroundingDinoForObjectDetection): Loaded Grounding DINO model.
        processor (AutoProcessor): Processor for preparing inputs to the model.

    Args:
        grounding_text (str): Text description of objects to detect (e.g., "dog", "red car").
        confidence_threshold (float): Threshold for filtering detection results. 
            Objects with confidence below this value will be discarded.
        model_config_path (str): Path to the pre-trained model configuration.
        device (str, optional): Device to run the model on. Defaults to "cpu".

    Example:
        >>> predictor = GroundingDinoPredictor(
        ...     grounding_text="cat",
        ...     confidence_threshold=0.5,
        ...     model_config_path="path/to/model/config"
        ... )
        >>> image = PIL.Image.open("example.jpg")
        >>> results = predictor.predict(image)
    """
    def __init__(self, 
                 grounding_text: str,
                 confidence_threshold: float, 
                 model_config_path: str,
                 device: str = "cpu"):
        
        self.grounding_text = grounding_text
        self.confidence_threshold = confidence_threshold
        self.config_path = model_config_path
        self.device = device
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_config_path)
        self.processor = AutoProcessor.from_pretrained(model_config_path)
        self.model.eval()
        self.model.to(self.device)
        
    @torch.inference_mode()    
    def predict(self, image: PIL.Image):
        inputs  = self.processor(images=image, text=self.grounding_text, return_tensors="pt").to(self.device)
        with torch.autocast(device_type="cuda"):
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
        )[0]
        return results

class CoilTracker:
    """
    A class for tracking the presence of coils in video frames using a detection model.
    
    This tracker maintains a buffer of detection results and provides a filtered status
    based on the recent history of detections. It includes functionality to filter out
    bounding boxes that are too large (likely false positives).
    
    Attributes:
        model: The detection model used to identify coils in frames.
        buffer_length (int): Maximum number of detection results to store in history.
        window_size (int): Number of recent results to consider when determining status.
        tracking_buffer (list): History of detection results (1 for detected, 0 for not detected).
        latest_bbox (list): The most recent valid bounding box detected.
        
    Methods:
        add(frame, is_coil_present=0): Add a detection result to the buffer, optionally running inference.
        get_status(): Return current tracking status based on recent detection history.
        _predict(frame): Run inference on a single frame to determine if a coil is present.
        _add_to_buffer(is_coil_present): Add a detection result to the tracking buffer.
        _filter_large_bboxes(bboxes, image_width, image_height, size_tolerance): Filter out bounding 
                                                                                 boxes that are too large.
    """
    def __init__(self, model=None, buffer_length=50, window_size=25):
        self.model = model
        self.buffer_length = buffer_length
        self.window_size = window_size
        self.tracking_buffer = []
        self.latest_bbox = None

    def _add_to_buffer(self, is_coil_present):
        if len(self.tracking_buffer) < self.buffer_length:
            self.tracking_buffer.append(is_coil_present)
        else:
            self.tracking_buffer.pop(0)
            self.tracking_buffer.append(is_coil_present)

    def _filter_large_bboxes(
        self, 
        bboxes, 
        image_width,
        image_height,
        size_tolerance=0.95
    ):
        """
        Eliminate bounding boxes that are approximately the same size as the original frame.
        
        Args:
        bboxes (list): List of bounding boxes, where each bbox is [x1, y1, x2, y2]
        frame_width (int or float): Width of the original frame
        frame_height (int or float): Height of the original frame
        size_tolerance (float): Tolerance for considering a box "similar" to frame size
                                Default is 0.95 (95% of frame size)
        
        Returns:
        list: Filtered list of bounding boxes
        """
        filtered_bboxes = []
        for bbox in bboxes:
            # Unpack bbox coordinates
            x1, y1, x2, y2 = bbox
            
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Calculate frame area and box area
            image_area = image_width * image_height
            box_area = box_width * box_height
            
            # Calculate relative areas and dimension ratios
            relative_area = box_area / image_area
            
            # Check if box is not close to frame size
            if (relative_area < size_tolerance):
                filtered_bboxes.append(bbox)
        return filtered_bboxes

    def _predict(self, frame):
        if isinstance(frame, ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        res = self.model.predict(frame)
        scores = res["scores"]
        is_coil_present = 0
        print("Scores:", scores)
        if scores.shape[0] > 0:
            max_score_idx = torch.argmax(scores)
            max_score_bbox = res["boxes"][max_score_idx].cpu().numpy().tolist()
            bboxes = self._filter_large_bboxes([max_score_bbox], frame.size[1], frame.size[0])
            is_coil_present = 1 if len(bboxes) else 0
            # drawn = draw_bboxes_with_labels(frame, {"coil": bboxes})
            # drawn.save("pred.jpg")
            if is_coil_present:
                self.latest_bbox = bboxes
        return is_coil_present

    def add(self, frame, is_coil_present=0):
        if self.model:
            is_coil_present = self._predict(frame)
        self._add_to_buffer(is_coil_present)

    def get_status(self):
        if sum(self.tracking_buffer) > (self.window_size - 1):
            return 1
        return 0

class Reader:
    """
    A class for performing OCR (Optical Character Recognition) using a pre-trained language model.
    
    This class loads a vision-language model capable of extracting text from images,
    and provides methods to process images and return the recognized text.
    
    Attributes:
        prompt (str): The prompt string used to instruct the model to perform OCR.
        device (str): The device to run inference on (e.g., "cpu", "cuda").
        model_path (str): Path to the pre-trained model.
        model: The loaded causal language model.
        processor: The processor for preparing inputs for the model.
        
    Methods:
        predict(image): Process an image and return the extracted text.
    """
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