import cv2
import numpy as np
from typing import Optional, Tuple, Any, Union
import threading

try:
    import mediapipe as mp
    from mediapipe.python.solutions.face_detection import FaceDetection
    from mediapipe.python.solutions.hands import Hands
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MediaPipe not available: {e}")
    print("Falling back to OpenCV tracking")
    MEDIAPIPE_AVAILABLE = False

class ObjectTracker:
    def __init__(self, tracking_mode: str = "face", min_confidence: float = 0.6, smoothing_factor: float = 0.3):
        # Input validation
        if tracking_mode not in ["face", "hand"]:
            raise ValueError("tracking_mode must be 'face' or 'hand'")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not 0.0 <= smoothing_factor <= 1.0:
            raise ValueError("smoothing_factor must be between 0.0 and 1.0")
        
        self.tracking_mode = tracking_mode
        self.min_confidence = min_confidence
        self.smoothing_factor = smoothing_factor
        self.detector: Optional[Any] = None
        self.detection_type: str = "none"
        
        # Thread safety
        self._lock = threading.RLock()
        
        if MEDIAPIPE_AVAILABLE:
            try:
                if tracking_mode == "face":
                    self.detector = FaceDetection(min_detection_confidence=min_confidence)
                    self.detection_type = "mediapipe_face"
                elif tracking_mode == "hand":
                    self.detector = Hands(min_detection_confidence=min_confidence, min_tracking_confidence=min_confidence)
                    self.detection_type = "mediapipe_hand"
            except Exception as e:
                print(f"MediaPipe initialization failed: {e}")
                self._init_opencv_fallback()
        else:
            self._init_opencv_fallback()
        
        # Thread-safe state initialization
        with self._lock:
            self.target_center: Optional[Tuple[float, float]] = None
            self.frame_offset = np.array([0.0, 0.0])
            self.target_offset = np.array([0.0, 0.0])
            self.is_tracking = False
            self.zoom_factor = 1.0
            self.max_offset = 0.3
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV fallback"""
        try:
            if self.tracking_mode == "face":
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                self.detection_type = "opencv_face"
            else:
                self.detector = None
                self.detection_type = "none"
                print("Hand tracking not available with OpenCV fallback")
        except AttributeError:
            print("OpenCV data module not available")
            self.detector = None
            self.detection_type = "none"
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        
        h, w = frame.shape[:2]
        detected = False
        
        if self.detection_type.startswith("mediapipe"):
            detected = self._process_mediapipe(frame, w, h)
        elif self.detection_type == "opencv_face":
            detected = self._process_opencv_detection(frame, w, h)
        
        # Thread-safe state updates
        with self._lock:
            if not detected:
                self.is_tracking = False
                self.target_offset *= 0.95
            
            self.frame_offset = (
                self.smoothing_factor * self.target_offset + 
                (1 - self.smoothing_factor) * self.frame_offset
            )
        
        transformed_frame = self._apply_virtual_camera(frame)
        if self.is_tracking and self.target_center is not None:
            transformed_frame = self._draw_tracking_info(transformed_frame)
        
        return transformed_frame
    
    def _process_mediapipe(self, frame: np.ndarray, w: int, h: int) -> bool:
        if self.detector is None:
            return False
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            
            if self.detection_type == "mediapipe_face":
                return self._process_mediapipe_face_detection(results, w, h)
            elif self.detection_type == "mediapipe_hand":
                return self._process_mediapipe_hand_detection(results, w, h)
        except Exception as e:
            print(f"MediaPipe processing error: {e}")
        
        return False
    
    def _process_mediapipe_face_detection(self, results: Any, w: int, h: int) -> bool:
        if hasattr(results, 'detections') and results.detections:
            best_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bbox = best_detection.location_data.relative_bounding_box
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            
            with self._lock:
                self.target_center = (center_x, center_y)
                self._update_target_offset(center_x, center_y)
                self.is_tracking = True
            return True
        return False
    
    def _process_mediapipe_hand_detection(self, results: Any, w: int, h: int) -> bool:
        if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Use wrist landmark (index 0) for hand center
            wrist = hand_landmarks.landmark[0]  # WRIST landmark
            center_x = wrist.x
            center_y = wrist.y
            
            with self._lock:
                self.target_center = (center_x, center_y)
                self._update_target_offset(center_x, center_y)
                self.is_tracking = True
            return True
        return False
    
    def _process_opencv_detection(self, frame: np.ndarray, w: int, h: int) -> bool:
        if self.detector is None:
            return False
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w_face, h_face = largest_face
                center_x = (x + w_face / 2) / w
                center_y = (y + h_face / 2) / h
                
                with self._lock:
                    self.target_center = (center_x, center_y)
                    self._update_target_offset(center_x, center_y)
                    self.is_tracking = True
                return True
        except Exception as e:
            print(f"OpenCV processing error: {e}")
        
        return False
    
    def _update_target_offset(self, center_x: float, center_y: float):
        """Thread-safe target offset update"""
        offset_x = center_x - 0.5
        offset_y = center_y - 0.5
        offset_x = np.clip(offset_x, -self.max_offset, self.max_offset)
        offset_y = np.clip(offset_y, -self.max_offset, self.max_offset)
        self.target_offset = np.array([-offset_x, -offset_y])
    
    def _apply_virtual_camera(self, frame: np.ndarray) -> np.ndarray:
        """Thread-safe virtual camera transformation"""
        h, w = frame.shape[:2]
        
        with self._lock:
            tx = int(self.frame_offset[0] * w)
            ty = int(self.frame_offset[1] * h)
            zoom_factor = self.zoom_factor
        
        # Create transformation matrix
        M = np.array([
            [zoom_factor, 0, tx],
            [0, zoom_factor, ty]
        ], dtype=np.float32)
        
        transformed = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return transformed
    
    def _draw_tracking_info(self, frame: np.ndarray) -> np.ndarray:
        """Thread-safe tracking info drawing"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        
        with self._lock:
            target_center = self.target_center
            frame_offset = self.frame_offset.copy()
        
        if target_center:
            target_x = int(target_center[0] * w)
            target_y = int(target_center[1] * h)
            cv2.circle(frame, (target_x, target_y), 10, (0, 0, 255), 2)
            cv2.circle(frame, (target_x, target_y), 3, (0, 0, 255), -1)
        
        status_text = f"Tracking: {self.tracking_mode.upper()}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        offset_text = f"Offset: ({frame_offset[0]:.2f}, {frame_offset[1]:.2f})"
        cv2.putText(frame, offset_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def set_zoom_factor(self, zoom: float):
        """Thread-safe zoom factor setting"""
        if not 0.5 <= zoom <= 3.0:
            raise ValueError("Zoom factor must be between 0.5 and 3.0")
        with self._lock:
            self.zoom_factor = zoom
    
    def set_smoothing_factor(self, factor: float):
        """Thread-safe smoothing factor setting"""
        if not 0.0 <= factor <= 1.0:
            raise ValueError("Smoothing factor must be between 0.0 and 1.0")
        with self._lock:
            self.smoothing_factor = factor
    
    def set_max_offset(self, offset: float):
        """Thread-safe max offset setting"""
        if not 0.1 <= offset <= 0.5:
            raise ValueError("Max offset must be between 0.1 and 0.5")
        with self._lock:
            self.max_offset = offset
    
    def reset_tracking(self):
        """Thread-safe tracking reset"""
        with self._lock:
            self.target_center = None
            self.frame_offset = np.array([0.0, 0.0])
            self.target_offset = np.array([0.0, 0.0])
            self.is_tracking = False
    
    def get_tracking_status(self) -> dict:
        """Thread-safe status retrieval"""
        with self._lock:
            return {
                'is_tracking': self.is_tracking,
                'target_center': self.target_center,
                'frame_offset': self.frame_offset.copy(),
                'zoom_factor': self.zoom_factor,
                'detection_type': self.detection_type
            }