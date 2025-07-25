import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Any, Dict
from utils.validators import InputValidator
from utils.error_handler import ErrorHandler, ProcessingError
from constants import *

try:
    import mediapipe as mp
    from mediapipe.python.solutions.face_detection import FaceDetection
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MediaPipe not available: {e}")
    print("Falling back to OpenCV face detection")
    MEDIAPIPE_AVAILABLE = False

class AutoFocus:
    """Auto-focus module with comprehensive error handling and performance optimization"""
    
    def __init__(self, 
                 min_confidence: float = AUTOFOCUS_DEFAULT_MIN_CONFIDENCE,
                 padding: int = AUTOFOCUS_DEFAULT_PADDING,
                 alpha: float = AUTOFOCUS_DEFAULT_ALPHA,
                 min_crop_ratio: float = AUTOFOCUS_DEFAULT_MIN_CROP_RATIO,
                 error_handler: Optional[ErrorHandler] = None):
        
        # Input validation using centralized validators
        self.min_confidence = InputValidator.validate_confidence(min_confidence, "min_confidence")
        self.padding = InputValidator.validate_positive_integer(padding, "padding")
        self.alpha = InputValidator.validate_alpha(alpha, "alpha")
        self.min_crop_ratio = InputValidator.validate_alpha(min_crop_ratio, "min_crop_ratio")
        
        # Error handling
        self.error_handler = error_handler or ErrorHandler()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # State management
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.detector: Optional[Any] = None
        self.detection_type: str = "none"
        
        # Performance tracking
        self.processing_times = []
        self.total_frames_processed = 0
        self.last_process_time = 0.0
        
        # Initialize detection
        self._init_detection()
    
    def _init_detection(self):
        """Initialize face detection with fallback"""
        if MEDIAPIPE_AVAILABLE:
            try:
                self.detector = FaceDetection(min_detection_confidence=self.min_confidence)
                self.detection_type = "mediapipe"
                self.error_handler.logger.info(LOG_MESSAGES['module_initialized'].format("MediaPipe AutoFocus"))
            except Exception as e:
                self.error_handler.logger.warning(LOG_MESSAGES['fallback_activated'].format("MediaPipe"))
                self._init_opencv_fallback()
        else:
            self._init_opencv_fallback()
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV fallback"""
        try:
            # Use getBuildInformation to check if OpenCV has data module
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.detection_type = "opencv"
            self.error_handler.logger.info(LOG_MESSAGES['module_initialized'].format("OpenCV AutoFocus"))
        except AttributeError:
            # Fallback if cv2.data is not available
            self.error_handler.logger.warning("OpenCV data module not available, using basic fallback")
            self.detector = None
            self.detection_type = "none"

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for auto-focus with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Input validation
            if frame is None or frame.size == 0:
                return frame
            
            # Validate image array
            InputValidator.validate_image_array(frame)
            
            ih, iw = frame.shape[:2]
            
            # Process based on detection type
            if self.detection_type == "mediapipe":
                processed_frame = self._process_mediapipe(frame, ih, iw)
            elif self.detection_type == "opencv":
                processed_frame = self._process_opencv(frame, ih, iw)
            else:
                processed_frame = frame
            
            # Update performance stats
            process_time = time.time() - start_time
            with self._lock:
                self.processing_times.append(process_time)
                self.total_frames_processed += 1
                self.last_process_time = process_time
                
                # Keep only last 100 processing times
                if len(self.processing_times) > PERFORMANCE_MAX_PROCESSING_TIMES:
                    self.processing_times.pop(0)
            
            return processed_frame
            
        except Exception as e:
            self.error_handler.handle_error(e, "autofocus_process")
            return frame

    def _process_mediapipe(self, frame: np.ndarray, ih: int, iw: int) -> np.ndarray:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            
            if hasattr(results, 'detections') and results.detections:
                best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                bbox = best.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                
                if self.prev_bbox:
                    x = int(self.alpha * x + (1 - self.alpha) * self.prev_bbox[0])
                    y = int(self.alpha * y + (1 - self.alpha) * self.prev_bbox[1])
                    w = int(self.alpha * w + (1 - self.alpha) * self.prev_bbox[2])
                    h = int(self.alpha * h + (1 - self.alpha) * self.prev_bbox[3])
                
                self.prev_bbox = (x, y, w, h)
            elif self.prev_bbox:
                x, y, w, h = self.prev_bbox
            else:
                return frame
            
            return self._apply_crop(frame, x, y, w, h, iw, ih)
        except Exception as e:
            print(f"MediaPipe processing error: {e}")
            return frame

    def _process_opencv(self, frame: np.ndarray, ih: int, iw: int) -> np.ndarray:
        if self.detector is None:
            return frame
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                if self.prev_bbox:
                    x = int(self.alpha * x + (1 - self.alpha) * self.prev_bbox[0])
                    y = int(self.alpha * y + (1 - self.alpha) * self.prev_bbox[1])
                    w = int(self.alpha * w + (1 - self.alpha) * self.prev_bbox[2])
                    h = int(self.alpha * h + (1 - self.alpha) * self.prev_bbox[3])
                
                self.prev_bbox = (x, y, w, h)
            elif self.prev_bbox:
                x, y, w, h = self.prev_bbox
            else:
                return frame
            
            return self._apply_crop(frame, x, y, w, h, iw, ih)
        except Exception as e:
            print(f"OpenCV processing error: {e}")
            return frame

    def _apply_crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int, iw: int, ih: int) -> np.ndarray:
        """Apply cropping with minimum size and padding constraints"""
        min_w, min_h = int(iw * self.min_crop_ratio), int(ih * self.min_crop_ratio)
        w = max(w, min_w)
        h = max(h, min_h)
        
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(iw, x + w + self.padding)
        y2 = min(ih, y + h + self.padding)
        
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame
        
        focused = cv2.resize(cropped, (iw, ih), interpolation=cv2.INTER_CUBIC)
        return focused
