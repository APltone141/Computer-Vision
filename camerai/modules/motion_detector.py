import cv2
import numpy as np
from collections import deque

class MotionDetector:
    def __init__(self, sensitivity=50, min_area=500, history_length=10):
        """
        Motion detector menggunakan background subtraction
        
        Args:
            sensitivity: threshold untuk deteksi motion (0-255)
            min_area: minimum area untuk dianggap sebagai motion
            history_length: jumlah frame untuk background learning
        """
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.history_length = history_length
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Motion state
        self.motion_detected = False
        self.motion_areas = []
        self.motion_history = deque(maxlen=history_length)
        
        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Statistics
        self.total_motion_area = 0
        self.motion_percentage = 0.0
        self.motion_intensity = 0.0
        
    def process(self, frame):
        """
        Process frame untuk motion detection
        """
        original_frame = frame.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply threshold
        _, thresh = cv2.threshold(fg_mask, self.sensitivity, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        self.motion_areas = []
        self.total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                self.motion_areas.append((x, y, w, h, area))
                self.total_motion_area += area
        
        # Update motion state
        self.motion_detected = len(self.motion_areas) > 0
        
        # Calculate motion statistics
        frame_area = frame.shape[0] * frame.shape[1]
        self.motion_percentage = (self.total_motion_area / frame_area) * 100
        
        # Motion intensity based on recent history
        self.motion_history.append(self.motion_percentage)
        self.motion_intensity = np.mean(self.motion_history) if self.motion_history else 0
        
        # Draw motion visualization
        output_frame = self._draw_motion_visualization(original_frame, thresh)
        
        return output_frame
    
    def _draw_motion_visualization(self, frame, motion_mask):
        """Draw motion detection visualization"""
        output = frame.copy()
        
        # Draw motion bounding boxes
        for x, y, w, h, area in self.motion_areas:
            # Color based on area size
            if area > self.min_area * 5:
                color = (0, 0, 255)  # Red for large motion
            elif area > self.min_area * 2:
                color = (0, 165, 255)  # Orange for medium motion
            else:
                color = (0, 255, 0)  # Green for small motion
            
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw area text
            cv2.putText(output, f"{int(area)}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw motion status
        status_color = (0, 255, 0) if self.motion_detected else (128, 128, 128)
        status_text = "MOTION DETECTED" if self.motion_detected else "NO MOTION"
        cv2.putText(output, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw statistics
        stats_text = f"Motion: {self.motion_percentage:.1f}% | Areas: {len(self.motion_areas)}"
        cv2.putText(output, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        intensity_text = f"Intensity: {self.motion_intensity:.1f}%"
        cv2.putText(output, intensity_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw motion heatmap overlay (optional)
        if self.motion_detected:
            # Create colored motion overlay
            motion_overlay = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
            output = cv2.addWeighted(output, 0.8, motion_overlay, 0.2, 0)
        
        return output
    
    def get_motion_areas(self):
        """Get current motion areas"""
        return self.motion_areas
    
    def get_motion_stats(self):
        """Get motion statistics"""
        return {
            'detected': self.motion_detected,
            'total_area': self.total_motion_area,
            'percentage': self.motion_percentage,
            'intensity': self.motion_intensity,
            'num_areas': len(self.motion_areas)
        }
    
    def set_sensitivity(self, sensitivity):
        """Set motion sensitivity threshold"""
        self.sensitivity = max(0, min(255, sensitivity))
    
    def set_min_area(self, min_area):
        """Set minimum area for motion detection"""
        self.min_area = max(10, min_area)
    
    def reset_background(self):
        """Reset background model"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        self.motion_history.clear()
    
    def set_learning_rate(self, rate):
        """Set background learning rate (0.0 = no learning, 1.0 = fast learning)"""
        # This affects how quickly the background model adapts
        pass  # MOG2 handles this automatically, but we can add manual control if needed


class MotionTrigger:
    """
    Motion-based trigger system untuk mengaktifkan fitur lain
    """
    def __init__(self, motion_detector, trigger_threshold=5.0, cooldown_time=30):
        """
        Args:
            motion_detector: MotionDetector instance
            trigger_threshold: percentage threshold untuk trigger
            cooldown_time: frames to wait before next trigger
        """
        self.motion_detector = motion_detector
        self.trigger_threshold = trigger_threshold
        self.cooldown_time = cooldown_time
        self.cooldown_counter = 0
        self.triggered = False
        
    def update(self):
        """Update trigger state"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        stats = self.motion_detector.get_motion_stats()
        
        if stats['percentage'] > self.trigger_threshold:
            self.triggered = True
            self.cooldown_counter = self.cooldown_time
            return True
        
        self.triggered = False
        return False
    
    def is_triggered(self):
        """Check if currently triggered"""
        return self.triggered
    
    def set_threshold(self, threshold):
        """Set trigger threshold"""
        self.trigger_threshold = max(0.1, min(100.0, threshold))