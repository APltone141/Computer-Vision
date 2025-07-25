"""
Input validation utilities for CameraAI application
Provides comprehensive validation for all input parameters to prevent vulnerabilities
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from constants import *

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class InputValidator:
    """Comprehensive input validation for CameraAI"""
    
    @staticmethod
    def validate_camera_index(camera_index: int) -> int:
        """Validate camera index"""
        if not isinstance(camera_index, int):
            raise ValidationError("Camera index must be integer")
        if camera_index < 0:
            raise ValidationError("Camera index must be non-negative")
        return camera_index
    
    @staticmethod
    def validate_resolution(width: int, height: int) -> Tuple[int, int]:
        """Validate resolution parameters"""
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValidationError("Width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValidationError("Width and height must be positive")
        if width > 10000 or height > 10000:
            raise ValidationError("Resolution too large (max 10000x10000)")
        return width, height
    
    @staticmethod
    def validate_fps(fps: float) -> float:
        """Validate FPS parameter"""
        if not isinstance(fps, (int, float)):
            raise ValidationError("FPS must be numeric")
        if fps < FPS_MIN or fps > FPS_MAX:
            raise ValidationError(f"FPS must be between {FPS_MIN} and {FPS_MAX}")
        return float(fps)
    
    @staticmethod
    def validate_confidence(confidence: float, param_name: str = "confidence") -> float:
        """Validate confidence parameter"""
        if not isinstance(confidence, (int, float)):
            raise ValidationError(f"{param_name} must be numeric")
        if confidence < CONFIDENCE_MIN or confidence > CONFIDENCE_MAX:
            raise ValidationError(f"{param_name} must be between {CONFIDENCE_MIN} and {CONFIDENCE_MAX}")
        return float(confidence)
    
    @staticmethod
    def validate_alpha(alpha: float, param_name: str = "alpha") -> float:
        """Validate alpha parameter"""
        if not isinstance(alpha, (int, float)):
            raise ValidationError(f"{param_name} must be numeric")
        if alpha < ALPHA_MIN or alpha > ALPHA_MAX:
            raise ValidationError(f"{param_name} must be between {ALPHA_MIN} and {ALPHA_MAX}")
        return float(alpha)
    
    @staticmethod
    def validate_smoothing_factor(smoothing: float) -> float:
        """Validate smoothing factor"""
        if not isinstance(smoothing, (int, float)):
            raise ValidationError("Smoothing factor must be numeric")
        if smoothing < SMOOTHING_MIN or smoothing > SMOOTHING_MAX:
            raise ValidationError(f"Smoothing factor must be between {SMOOTHING_MIN} and {SMOOTHING_MAX}")
        return float(smoothing)
    
    @staticmethod
    def validate_scale_factor(scale: float) -> float:
        """Validate scale factor"""
        if not isinstance(scale, (int, float)):
            raise ValidationError("Scale factor must be numeric")
        if scale < SUPER_RESOLUTION_MIN_SCALE or scale > SUPER_RESOLUTION_MAX_SCALE:
            raise ValidationError(f"Scale factor must be between {SUPER_RESOLUTION_MIN_SCALE} and {SUPER_RESOLUTION_MAX_SCALE}")
        return float(scale)
    
    @staticmethod
    def validate_tracking_mode(mode: str) -> str:
        """Validate tracking mode"""
        if not isinstance(mode, str):
            raise ValidationError("Tracking mode must be string")
        if mode not in SUPPORTED_TRACKING_MODES:
            raise ValidationError(f"Tracking mode must be one of {SUPPORTED_TRACKING_MODES}")
        return mode
    
    @staticmethod
    def validate_interpolation_method(method: str) -> str:
        """Validate interpolation method"""
        if not isinstance(method, str):
            raise ValidationError("Interpolation method must be string")
        if method not in SUPPORTED_INTERPOLATION_METHODS:
            raise ValidationError(f"Interpolation method must be one of {SUPPORTED_INTERPOLATION_METHODS}")
        return method
    
    @staticmethod
    def validate_video_format(format: str) -> str:
        """Validate video format"""
        if not isinstance(format, str):
            raise ValidationError("Video format must be string")
        if format not in SUPPORTED_VIDEO_FORMATS:
            raise ValidationError(f"Video format must be one of {SUPPORTED_VIDEO_FORMATS}")
        return format
    
    @staticmethod
    def validate_video_codec(codec: str) -> str:
        """Validate video codec"""
        if not isinstance(codec, str):
            raise ValidationError("Video codec must be string")
        if codec not in SUPPORTED_VIDEO_CODECS:
            raise ValidationError(f"Video codec must be one of {SUPPORTED_VIDEO_CODECS}")
        return codec
    
    @staticmethod
    def validate_quality_mode(quality: str) -> str:
        """Validate quality mode"""
        if not isinstance(quality, str):
            raise ValidationError("Quality mode must be string")
        if quality not in SUPPORTED_QUALITY_MODES:
            raise ValidationError(f"Quality mode must be one of {SUPPORTED_QUALITY_MODES}")
        return quality
    
    @staticmethod
    def validate_file_path(filepath: str, must_exist: bool = False) -> str:
        """Validate file path"""
        if not isinstance(filepath, str):
            raise ValidationError("File path must be string")
        
        # Check for path traversal attacks
        if ".." in filepath or "//" in filepath:
            raise ValidationError("Invalid file path")
        
        # Check for absolute paths (optional security measure)
        if os.path.isabs(filepath):
            # Allow absolute paths but log warning
            pass
        
        if must_exist and not os.path.exists(filepath):
            raise ValidationError(f"File does not exist: {filepath}")
        
        return filepath
    
    @staticmethod
    def validate_directory_path(dirpath: str, create_if_missing: bool = False) -> str:
        """Validate directory path"""
        if not isinstance(dirpath, str):
            raise ValidationError("Directory path must be string")
        
        # Check for path traversal attacks
        if ".." in dirpath or "//" in dirpath:
            raise ValidationError("Invalid directory path")
        
        if create_if_missing:
            os.makedirs(dirpath, exist_ok=True)
        elif not os.path.exists(dirpath):
            raise ValidationError(f"Directory does not exist: {dirpath}")
        
        return dirpath
    
    @staticmethod
    def validate_image_array(image: np.ndarray) -> np.ndarray:
        """Validate image array"""
        if not isinstance(image, np.ndarray):
            raise ValidationError("Image must be numpy array")
        
        if image.size == 0:
            raise ValidationError("Image array is empty")
        
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValidationError("Image must be 2D or 3D array")
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValidationError("Image must have 1, 3, or 4 channels")
        
        # Check for invalid values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            raise ValidationError("Image contains invalid values (NaN or Inf)")
        
        return image
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary"""
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be dictionary")
        
        # Validate required keys
        required_keys = ['camera', 'modules', 'gui', 'performance']
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required configuration key: {key}")
        
        # Validate camera configuration
        camera_config = config.get('camera', {})
        if not isinstance(camera_config, dict):
            raise ValidationError("Camera configuration must be dictionary")
        
        # Validate modules configuration
        modules_config = config.get('modules', {})
        if not isinstance(modules_config, dict):
            raise ValidationError("Modules configuration must be dictionary")
        
        return config
    
    @staticmethod
    def validate_gesture_name(gesture: str) -> str:
        """Validate gesture name"""
        if not isinstance(gesture, str):
            raise ValidationError("Gesture name must be string")
        
        if gesture not in GESTURE_DEFINITIONS:
            raise ValidationError(f"Invalid gesture name: {gesture}")
        
        return gesture
    
    @staticmethod
    def validate_model_name(model: str) -> str:
        """Validate AI model name"""
        if not isinstance(model, str):
            raise ValidationError("Model name must be string")
        
        if model not in SUPER_RESOLUTION_MODELS:
            raise ValidationError(f"Invalid model name: {model}")
        
        return model
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            raise ValidationError("Input must be string")
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_str)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_positive_integer(value: int, param_name: str = "value") -> int:
        """Validate positive integer"""
        if not isinstance(value, int):
            raise ValidationError(f"{param_name} must be integer")
        if value <= 0:
            raise ValidationError(f"{param_name} must be positive")
        return value
    
    @staticmethod
    def validate_non_negative_integer(value: int, param_name: str = "value") -> int:
        """Validate non-negative integer"""
        if not isinstance(value, int):
            raise ValidationError(f"{param_name} must be integer")
        if value < 0:
            raise ValidationError(f"{param_name} must be non-negative")
        return value

class SecurityValidator:
    """Security-focused validation utilities"""
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> str:
        """Validate file extension for security"""
        if not isinstance(filename, str):
            raise ValidationError("Filename must be string")
        
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValidationError("Invalid filename")
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValidationError(f"File extension not allowed: {file_ext}")
        
        return filename
    
    @staticmethod
    def validate_url(url: str) -> str:
        """Validate URL for security"""
        if not isinstance(url, str):
            raise ValidationError("URL must be string")
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            raise ValidationError("Invalid URL format")
        
        return url
    
    @staticmethod
    def validate_json_data(data: str) -> str:
        """Validate JSON data for security"""
        if not isinstance(data, str):
            raise ValidationError("Data must be string")
        
        # Check for potential injection
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                raise ValidationError("Potentially dangerous content detected")
        
        return data

class PerformanceValidator:
    """Performance-focused validation utilities"""
    
    @staticmethod
    def validate_memory_limit(memory_mb: float) -> float:
        """Validate memory limit"""
        if not isinstance(memory_mb, (int, float)):
            raise ValidationError("Memory limit must be numeric")
        if memory_mb <= 0:
            raise ValidationError("Memory limit must be positive")
        if memory_mb > PERFORMANCE_MAX_MEMORY_USAGE:
            raise ValidationError(f"Memory limit too high (max {PERFORMANCE_MAX_MEMORY_USAGE}MB)")
        return float(memory_mb)
    
    @staticmethod
    def validate_cpu_limit(cpu_percent: float) -> float:
        """Validate CPU limit"""
        if not isinstance(cpu_percent, (int, float)):
            raise ValidationError("CPU limit must be numeric")
        if cpu_percent <= 0 or cpu_percent > 100:
            raise ValidationError("CPU limit must be between 0 and 100")
        return float(cpu_percent)
    
    @staticmethod
    def validate_processing_timeout(timeout: float) -> float:
        """Validate processing timeout"""
        if not isinstance(timeout, (int, float)):
            raise ValidationError("Timeout must be numeric")
        if timeout <= 0:
            raise ValidationError("Timeout must be positive")
        if timeout > 60:  # Max 60 seconds
            raise ValidationError("Timeout too high (max 60 seconds)")
        return float(timeout) 