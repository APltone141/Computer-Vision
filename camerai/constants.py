"""
Constants and configuration values for CameraAI application
Centralized configuration to eliminate magic numbers and improve maintainability
"""

# Camera Configuration
CAMERA_DEFAULT_INDEX = 0
CAMERA_DEFAULT_WIDTH = 640
CAMERA_DEFAULT_HEIGHT = 480
CAMERA_DEFAULT_FPS = 30
CAMERA_DEFAULT_QUEUE_SIZE = 64

# Performance Configuration
PERFORMANCE_MAX_FPS = 60
PERFORMANCE_MAX_WORKERS = 3
PERFORMANCE_HISTORY_SIZE = 100
PERFORMANCE_MAX_PROCESSING_TIMES = 100
PERFORMANCE_MAX_ERROR_LOGS = 100
PERFORMANCE_MAX_PERFORMANCE_LOGS = 1000

# Module Configuration
AUTOFOCUS_DEFAULT_MIN_CONFIDENCE = 0.6
AUTOFOCUS_DEFAULT_PADDING = 50
AUTOFOCUS_DEFAULT_ALPHA = 0.4
AUTOFOCUS_DEFAULT_MIN_CROP_RATIO = 0.6

TRACKING_DEFAULT_MIN_CONFIDENCE = 0.6
TRACKING_DEFAULT_SMOOTHING_FACTOR = 0.3
TRACKING_DEFAULT_ZOOM_FACTOR = 1.0
TRACKING_DEFAULT_MAX_OFFSET = 0.3

MOTION_DETECTION_DEFAULT_SENSITIVITY = 50
MOTION_DETECTION_DEFAULT_MIN_AREA = 500
MOTION_DETECTION_DEFAULT_HISTORY_LENGTH = 10

GESTURE_CONTROL_DEFAULT_MIN_DETECTION_CONFIDENCE = 0.7
GESTURE_CONTROL_DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5
GESTURE_CONTROL_DEFAULT_COOLDOWN = 1.0

SUPER_RESOLUTION_DEFAULT_SCALE_FACTOR = 2.0
SUPER_RESOLUTION_MIN_SCALE = 1.0
SUPER_RESOLUTION_MAX_SCALE = 4.0

FPS_INTERPOLATOR_DEFAULT_TARGET_FPS = 60.0
FPS_INTERPOLATOR_DEFAULT_FRAME_BUFFER_SIZE = 10

# GUI Configuration
GUI_DEFAULT_WINDOW_WIDTH = 1200
GUI_DEFAULT_WINDOW_HEIGHT = 800
GUI_DEFAULT_THEME = "default"

# Recording Configuration
RECORDING_DEFAULT_FPS = 30.0
RECORDING_DEFAULT_RESOLUTION = (640, 480)
RECORDING_DEFAULT_QUALITY = "high"
RECORDING_DEFAULT_FORMAT = "mp4"
RECORDING_DEFAULT_CODEC = "H264"

# Logger Configuration
LOGGER_DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOGGER_DEFAULT_BACKUP_COUNT = 5
LOGGER_DEFAULT_LOG_LEVEL = "INFO"

# Error Handling Configuration
ERROR_MAX_CONSECUTIVE_ERRORS = 5
ERROR_MAX_CAMERA_ERRORS = 10
ERROR_TIMEOUT_SECONDS = 1.0
ERROR_RETRY_DELAY = 0.1

# Thread Safety Configuration
THREAD_LOCK_TIMEOUT = 1.0
THREAD_JOIN_TIMEOUT = 2.0

# Image Processing Configuration
IMAGE_PROCESSING_DEFAULT_QUALITY = 95
IMAGE_PROCESSING_DEFAULT_COMPRESSION = 9

# Validation Ranges
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0
SMOOTHING_MIN = 0.0
SMOOTHING_MAX = 1.0
FPS_MIN = 1.0
FPS_MAX = 300.0

# Supported Formats and Codecs
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]
SUPPORTED_VIDEO_CODECS = ["H264", "H265", "XVID", "MJPG"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_QUALITY_MODES = ["low", "medium", "high", "ultra"]
SUPPORTED_TRACKING_MODES = ["face", "hand"]
SUPPORTED_INTERPOLATION_METHODS = ["opencv", "linear", "cubic", "advanced"]

# Quality Settings
QUALITY_SETTINGS = {
    'low': {'crf': 28, 'preset': 'fast'},
    'medium': {'crf': 23, 'preset': 'medium'},
    'high': {'crf': 18, 'preset': 'slow'},
    'ultra': {'crf': 12, 'preset': 'veryslow'}
}

# Codec Mappings
CODEC_MAPPINGS = {
    'H264': 'H264',
    'H265': 'H265',
    'XVID': 'XVID',
    'MJPG': 'MJPG'
}

# Gesture Definitions
GESTURE_DEFINITIONS = {
    "fist": "fist",
    "open_palm": "open_palm",
    "pointing": "pointing",
    "thumbs_up": "thumbs_up",
    "thumbs_down": "thumbs_down",
    "peace": "peace",
    "ok": "ok",
    "rock_on": "rock_on"
}

# Command Mappings
COMMAND_MAPPINGS = {
    "fist": "toggle_autofocus",
    "open_palm": "toggle_tracking",
    "pointing": "toggle_motion",
    "thumbs_up": "increase_zoom",
    "thumbs_down": "decrease_zoom",
    "peace": "take_screenshot",
    "ok": "start_recording",
    "rock_on": "stop_recording"
}

# AI Model Names
SUPER_RESOLUTION_MODELS = [
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3"
]

# File Paths
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_LOG_DIR = "logs"
DEFAULT_RECORDING_DIR = "recordings"
DEFAULT_SCREENSHOT_DIR = "screenshots"
DEFAULT_ASSETS_DIR = "assets"

# Memory Management
MEMORY_LIMIT_MB = 100 * 1024 * 1024  # 100MB
FRAME_BUFFER_SIZE = 10
PROCESSING_TIMEOUT = 1.0

# Performance Thresholds
PERFORMANCE_TARGET_FPS = 30.0
PERFORMANCE_MIN_FPS = 15.0
PERFORMANCE_MAX_CPU_USAGE = 80.0
PERFORMANCE_MAX_MEMORY_USAGE = 1000.0  # MB

# Error Messages
ERROR_MESSAGES = {
    'camera_init_failed': "Camera initialization failed: {}",
    'camera_not_responding': "Camera {} is not responding",
    'invalid_parameter': "Invalid parameter: {}",
    'processing_error': "Processing error: {}",
    'thread_safety_error': "Thread safety error: {}",
    'memory_error': "Memory usage exceeded limit",
    'timeout_error': "Operation timed out",
    'validation_error': "Validation error: {}"
}

# Success Messages
SUCCESS_MESSAGES = {
    'camera_started': "Camera started successfully",
    'processing_complete': "Processing completed successfully",
    'configuration_saved': "Configuration saved successfully",
    'recording_started': "Recording started successfully",
    'recording_stopped': "Recording stopped successfully"
}

# Log Messages
LOG_MESSAGES = {
    'startup': "CameraAI application started",
    'shutdown': "CameraAI application shutting down",
    'module_initialized': "Module {} initialized successfully",
    'fallback_activated': "Fallback activated for {}",
    'performance_warning': "Performance warning: {}",
    'error_recovered': "Error recovered: {}"
} 