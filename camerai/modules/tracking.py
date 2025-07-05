import cv2
import mediapipe as mp
import numpy as np

class ObjectTracker:
    def __init__(self, tracking_mode="face", min_confidence=0.6, smoothing_factor=0.3):
        """
        Object tracker untuk face/hand tracking dengan smooth following
        
        Args:
            tracking_mode: "face" atau "hand"
            min_confidence: threshold confidence untuk deteksi
            smoothing_factor: faktor smoothing untuk pergerakan (0-1)
        """
        self.tracking_mode = tracking_mode
        self.min_confidence = min_confidence
        self.smoothing_factor = smoothing_factor
        
        # Initialize MediaPipe
        if tracking_mode == "face":
            self.detector = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=min_confidence
            )
        elif tracking_mode == "hand":
            self.detector = mp.solutions.hands.Hands(
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence
            )
        
        # Tracking state
        self.target_center = None
        self.frame_offset = np.array([0.0, 0.0])  # Current virtual camera offset
        self.target_offset = np.array([0.0, 0.0])  # Target offset to move to
        self.is_tracking = False
        
        # Virtual camera parameters
        self.zoom_factor = 1.0
        self.max_offset = 0.3  # Maximum offset as ratio of frame size
        
    def process(self, frame):
        """
        Process frame untuk tracking dan return transformed frame
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        # Reset tracking if no detection
        detected = False
        
        if self.tracking_mode == "face":
            detected = self._process_face_detection(results, w, h)
        elif self.tracking_mode == "hand":
            detected = self._process_hand_detection(results, w, h)
        
        if not detected:
            self.is_tracking = False
            # Gradually return to center when no target
            self.target_offset *= 0.95
        
        # Smooth movement towards target
        self.frame_offset = (
            self.smoothing_factor * self.target_offset + 
            (1 - self.smoothing_factor) * self.frame_offset
        )
        
        # Apply virtual camera transformation
        transformed_frame = self._apply_virtual_camera(frame)
        
        # Draw tracking info
        if self.is_tracking and self.target_center is not None:
            transformed_frame = self._draw_tracking_info(transformed_frame)
        
        return transformed_frame
    
    def _process_face_detection(self, results, w, h):
        """Process face detection results"""
        if not results.detections:
            return False
        
        # Get largest face
        best_detection = max(results.detections, 
                           key=lambda d: d.location_data.relative_bounding_box.width * 
                                       d.location_data.relative_bounding_box.height)
        
        bbox = best_detection.location_data.relative_bounding_box
        center_x = bbox.xmin + bbox.width / 2
        center_y = bbox.ymin + bbox.height / 2
        
        self.target_center = (center_x, center_y)
        self._update_target_offset(center_x, center_y)
        self.is_tracking = True
        
        return True
    
    def _process_hand_detection(self, results, w, h):
        """Process hand detection results"""
        if not results.multi_hand_landmarks:
            return False
        
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get hand center (wrist landmark)
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        center_x = wrist.x
        center_y = wrist.y
        
        self.target_center = (center_x, center_y)
        self._update_target_offset(center_x, center_y)
        self.is_tracking = True
        
        return True
    
    def _update_target_offset(self, center_x, center_y):
        """Update target offset based on detected object center"""
        # Calculate offset from frame center
        offset_x = center_x - 0.5
        offset_y = center_y - 0.5
        
        # Clamp offset
        offset_x = np.clip(offset_x, -self.max_offset, self.max_offset)
        offset_y = np.clip(offset_y, -self.max_offset, self.max_offset)
        
        # Set target offset (negative because we move frame opposite to object)
        self.target_offset = np.array([-offset_x, -offset_y])
    
    def _apply_virtual_camera(self, frame):
        """Apply virtual camera transformation (pan + zoom)"""
        h, w = frame.shape[:2]
        
        # Calculate translation
        tx = int(self.frame_offset[0] * w)
        ty = int(self.frame_offset[1] * h)
        
        # Create transformation matrix
        M = np.float32([
            [self.zoom_factor, 0, tx],
            [0, self.zoom_factor, ty]
        ])
        
        # Apply transformation
        transformed = cv2.warpAffine(frame, M, (w, h), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
        
        return transformed
    
    def _draw_tracking_info(self, frame):
        """Draw tracking visualization"""
        h, w = frame.shape[:2]
        
        # Draw center crosshair
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), 
                (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), 
                (0, 255, 0), 2)
        
        # Draw target position
        if self.target_center:
            target_x = int(self.target_center[0] * w)
            target_y = int(self.target_center[1] * h)
            cv2.circle(frame, (target_x, target_y), 10, (0, 0, 255), 2)
            cv2.circle(frame, (target_x, target_y), 3, (0, 0, 255), -1)
        
        # Draw tracking status
        status_text = f"Tracking: {self.tracking_mode.upper()}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw offset info
        offset_text = f"Offset: ({self.frame_offset[0]:.2f}, {self.frame_offset[1]:.2f})"
        cv2.putText(frame, offset_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def set_zoom_factor(self, zoom):
        """Set zoom factor (1.0 = normal, >1.0 = zoom in)"""
        self.zoom_factor = max(0.5, min(3.0, zoom))
    
    def set_smoothing_factor(self, factor):
        """Set smoothing factor (0 = no smoothing, 1 = max smoothing)"""
        self.smoothing_factor = max(0.0, min(1.0, factor))
    
    def set_max_offset(self, offset):
        """Set maximum offset ratio"""
        self.max_offset = max(0.1, min(0.5, offset))
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.target_center = None
        self.frame_offset = np.array([0.0, 0.0])
        self.target_offset = np.array([0.0, 0.0])
        self.is_tracking = False