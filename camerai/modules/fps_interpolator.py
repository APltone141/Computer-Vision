import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import threading
import time
from collections import deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available for advanced interpolation")
    TORCH_AVAILABLE = False

class FPSInterpolator:
    """
    FPS Interpolator untuk meningkatkan frame rate video
    Menggunakan frame interpolation dengan berbagai algoritma
    """
    
    def __init__(self, target_fps: float = 60.0, interpolation_method: str = "opencv"):
        # Input validation
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if interpolation_method not in ["opencv", "linear", "cubic", "advanced"]:
            raise ValueError("Invalid interpolation_method")
        
        self.target_fps = target_fps
        self.interpolation_method = interpolation_method
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Frame buffer
        with self._lock:
            self.frame_buffer = deque(maxlen=10)
            self.last_frame_time = 0.0
            self.frame_interval = 1.0 / target_fps
            self.interpolation_factor = 1.0
        
        # Performance tracking
        self.processing_times = []
        self.total_frames_generated = 0
        self.last_process_time = 0.0
        
        # Initialize interpolation method
        self._init_interpolation()
    
    def _init_interpolation(self):
        """Initialize interpolation method"""
        if self.interpolation_method == "advanced" and TORCH_AVAILABLE:
            self._init_advanced_interpolation()
        else:
            self._init_basic_interpolation()
    
    def _init_advanced_interpolation(self):
        """Initialize advanced interpolation with PyTorch"""
        try:
            # This would be where we'd initialize RIFE or DAIN models
            # For now, we'll use a simplified version
            self.interpolation_type = "advanced"
            print("✅ Advanced interpolation initialized")
        except Exception as e:
            print(f"Advanced interpolation failed: {e}")
            self._init_basic_interpolation()
    
    def _init_basic_interpolation(self):
        """Initialize basic interpolation methods"""
        self.interpolation_type = "basic"
        print(f"✅ Basic interpolation initialized: {self.interpolation_method}")
    
    def process(self, frame: np.ndarray, current_fps: float) -> List[np.ndarray]:
        """
        Process frame untuk FPS interpolation
        
        Args:
            frame: Input frame
            current_fps: Current FPS of input
            
        Returns:
            List of interpolated frames
        """
        if frame is None or frame.size == 0:
            return [frame]
        
        start_time = time.time()
        
        try:
            # Add frame to buffer
            with self._lock:
                self.frame_buffer.append(frame.copy())
            
            # Calculate how many interpolated frames we need
            if current_fps <= 0:
                return [frame]
            
            current_interval = 1.0 / current_fps
            target_interval = 1.0 / self.target_fps
            
            # If we need to increase FPS
            if target_interval < current_interval:
                frames_needed = int(current_interval / target_interval)
                interpolated_frames = self._interpolate_frames(frame, frames_needed)
            else:
                # If target FPS is lower, just return original frame
                interpolated_frames = [frame]
            
            # Update performance stats
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            self.total_frames_generated += len(interpolated_frames)
            self.last_process_time = process_time
            
            # Keep only last 100 processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return interpolated_frames
            
        except Exception as e:
            print(f"FPS interpolation error: {e}")
            return [frame]
    
    def _interpolate_frames(self, current_frame: np.ndarray, num_frames: int) -> List[np.ndarray]:
        """Generate interpolated frames"""
        if num_frames <= 0:
            return [current_frame]
        
        interpolated_frames = []
        
        # Get previous frame from buffer
        with self._lock:
            if len(self.frame_buffer) < 2:
                return [current_frame]
            prev_frame = self.frame_buffer[-2]
        
        # Generate interpolated frames
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            
            if self.interpolation_method == "linear":
                interpolated = self._linear_interpolation(prev_frame, current_frame, alpha)
            elif self.interpolation_method == "cubic":
                interpolated = self._cubic_interpolation(prev_frame, current_frame, alpha)
            elif self.interpolation_method == "advanced":
                interpolated = self._advanced_interpolation(prev_frame, current_frame, alpha)
            else:  # opencv
                interpolated = self._opencv_interpolation(prev_frame, current_frame, alpha)
            
            interpolated_frames.append(interpolated)
        
        return interpolated_frames
    
    def _linear_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Linear interpolation between two frames"""
        return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
    
    def _cubic_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Cubic interpolation between two frames"""
        # Convert to float for better precision
        f1 = frame1.astype(np.float32)
        f2 = frame2.astype(np.float32)
        
        # Cubic interpolation
        t = alpha
        t2 = t * t
        t3 = t2 * t
        
        # Cubic weights
        w1 = 2 * t3 - 3 * t2 + 1
        w2 = -2 * t3 + 3 * t2
        
        interpolated = w1 * f1 + w2 * f2
        
        return np.clip(interpolated, 0, 255).astype(np.uint8)
    
    def _opencv_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """OpenCV-based interpolation"""
        # Use optical flow for better interpolation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Create interpolated frame
        h, w = frame1.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow
        flow_x = x_coords + flow[:, :, 0] * alpha
        flow_y = y_coords + flow[:, :, 1] * alpha
        
        # Remap frame
        interpolated = cv2.remap(frame1, flow_x, flow_y, cv2.INTER_LINEAR)
        
        return interpolated
    
    def _advanced_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Advanced interpolation using multiple techniques"""
        if TORCH_AVAILABLE:
            # This would use RIFE or DAIN models
            # For now, use enhanced optical flow
            return self._enhanced_optical_flow(frame1, frame2, alpha)
        else:
            return self._cubic_interpolation(frame1, frame2, alpha)
    
    def _enhanced_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Enhanced optical flow interpolation"""
        # Convert to grayscale for flow calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate bidirectional optical flow
        flow_forward = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_backward = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Create interpolated frame
        h, w = frame1.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Combine forward and backward flow
        flow_x = x_coords + (flow_forward[:, :, 0] * alpha + flow_backward[:, :, 0] * (1 - alpha)) / 2
        flow_y = y_coords + (flow_forward[:, :, 1] * alpha + flow_backward[:, :, 1] * (1 - alpha)) / 2
        
        # Remap frame
        interpolated = cv2.remap(frame1, flow_x, flow_y, cv2.INTER_CUBIC)
        
        return interpolated
    
    def set_target_fps(self, fps: float):
        """Set target FPS"""
        if fps <= 0:
            raise ValueError("FPS must be positive")
        
        with self._lock:
            self.target_fps = fps
            self.frame_interval = 1.0 / fps
    
    def set_interpolation_method(self, method: str):
        """Set interpolation method"""
        if method not in ["opencv", "linear", "cubic", "advanced"]:
            raise ValueError("Invalid interpolation method")
        
        with self._lock:
            self.interpolation_method = method
        
        # Reinitialize interpolation
        self._init_interpolation()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_frames': 0,
                'target_fps': self.target_fps,
                'method': self.interpolation_method
            }
        
        return {
            'avg_time': np.mean(self.processing_times),
            'min_time': np.min(self.processing_times),
            'max_time': np.max(self.processing_times),
            'total_frames': self.total_frames_generated,
            'target_fps': self.target_fps,
            'method': self.interpolation_method,
            'last_process_time': self.last_process_time
        }
    
    def is_available(self) -> bool:
        """Check if FPS interpolation is available"""
        return True  # Always available with basic methods
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get interpolation method information"""
        return {
            'interpolation_method': self.interpolation_method,
            'interpolation_type': self.interpolation_type,
            'target_fps': self.target_fps,
            'torch_available': TORCH_AVAILABLE
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.processing_times.clear()
        self.total_frames_generated = 0
        self.last_process_time = 0.0
    
    def clear_buffer(self):
        """Clear frame buffer"""
        with self._lock:
            self.frame_buffer.clear()
    
    def optimize_for_quality(self):
        """Optimize for quality over speed"""
        if self.interpolation_method == "opencv":
            self.interpolation_method = "cubic"
        elif self.interpolation_method == "linear":
            self.interpolation_method = "cubic"
        
        self._init_interpolation()
    
    def optimize_for_speed(self):
        """Optimize for speed over quality"""
        if self.interpolation_method in ["cubic", "advanced"]:
            self.interpolation_method = "linear"
        
        self._init_interpolation()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.clear_buffer()
