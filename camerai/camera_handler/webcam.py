import cv2
import threading
from threading import Thread, Event
from queue import Queue
import time
from typing import Optional, Dict, Any
from utils.validators import InputValidator
from utils.error_handler import ErrorHandler, CameraError
from constants import *

class FrameGrabber:
    """Thread-safe frame grabber with comprehensive error handling"""
    
    def __init__(self, 
                 src: int = CAMERA_DEFAULT_INDEX,
                 width: int = CAMERA_DEFAULT_WIDTH,
                 height: int = CAMERA_DEFAULT_HEIGHT,
                 queue_size: int = CAMERA_DEFAULT_QUEUE_SIZE,
                 error_handler: Optional[ErrorHandler] = None):
        
        # Input validation using centralized validators
        self.src = InputValidator.validate_camera_index(src)
        self.width, self.height = InputValidator.validate_resolution(width, height)
        self.queue_size = InputValidator.validate_positive_integer(queue_size, "queue_size")
        
        # Error handling
        self.error_handler = error_handler or ErrorHandler()
        
        # Camera state
        self.cap = None
        self._capture_thread = None
        self.stopped = Event()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.consecutive_errors = 0
        
        # Initialize camera
        self._init_camera()
        
        # Frame queue
        self.queue = Queue(maxsize=self.queue_size)

    def _init_camera(self):
        """Initialize camera with comprehensive error handling"""
        try:
            with self._lock:
                self.cap = cv2.VideoCapture(self.src)
                if not self.cap.isOpened():
                    raise CameraError(ERROR_MESSAGES['camera_init_failed'].format(self.src))
                
                # Set camera properties with validation
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Test camera by reading one frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    raise CameraError(ERROR_MESSAGES['camera_not_responding'].format(self.src))
                
                self.error_handler.logger.info(SUCCESS_MESSAGES['camera_started'])
                
        except Exception as e:
            self._cleanup_camera()
            self.error_handler.handle_error(e, "camera_initialization")
            raise CameraError(ERROR_MESSAGES['camera_init_failed'].format(str(e)))
    
    def _cleanup_camera(self):
        """Clean up camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start(self):
        """Start frame capture thread with error handling"""
        try:
            with self._lock:
                if self.cap is None:
                    raise CameraError("Camera not initialized")
                
                if self._capture_thread is not None and self._capture_thread.is_alive():
                    return self  # Already running
                
                self.stopped.clear()
                self._capture_thread = Thread(target=self._capture_loop, daemon=True)
                self._capture_thread.start()
                
                self.error_handler.logger.info("Frame capture started")
                return self
                
        except Exception as e:
            self.error_handler.handle_error(e, "camera_start")
            raise

    def _capture_loop(self):
        """Main capture loop with comprehensive error handling"""
        while not self.stopped.is_set():
            try:
                with self._lock:
                    if self.cap is None:
                        break
                    
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self.consecutive_errors += 1
                        if self.consecutive_errors >= ERROR_MAX_CAMERA_ERRORS:
                            self.error_handler.logger.error(
                                ERROR_MESSAGES['camera_not_responding'].format(self.src)
                            )
                            break
                        time.sleep(ERROR_RETRY_DELAY)
                        continue
                    
                    self.consecutive_errors = 0  # Reset error counter
                    self.frame_count += 1
                    
                    # Queue management with timeout
                    self._queue_frame(frame)
                    
            except Exception as e:
                self.consecutive_errors += 1
                self.error_handler.handle_error(e, "capture_loop")
                
                if self.consecutive_errors >= ERROR_MAX_CAMERA_ERRORS:
                    break
                time.sleep(ERROR_RETRY_DELAY)
    
    def _queue_frame(self, frame):
        """Queue frame with proper error handling"""
        try:
            if self.queue.full():
                # Drop oldest frame to prevent memory issues
                try:
                    self.queue.get_nowait()
                except:
                    pass
            
            self.queue.put(frame, timeout=ERROR_TIMEOUT_SECONDS)
            
        except Exception as e:
            # Queue is full or timeout, skip this frame
            self.error_handler.logger.debug(f"Frame queue error: {e}")
            pass

    def read(self) -> Optional[cv2.Mat]:
        """Read frame from queue with timeout and error handling"""
        try:
            return self.queue.get(timeout=ERROR_TIMEOUT_SECONDS)
        except Exception as e:
            self.error_handler.logger.debug(f"Frame read timeout: {e}")
            return None

    def stop(self):
        """Stop capture and release resources with proper cleanup"""
        try:
            with self._lock:
                self.stopped.set()
                
                # Wait for thread to finish
                if self._capture_thread and self._capture_thread.is_alive():
                    self._capture_thread.join(timeout=THREAD_JOIN_TIMEOUT)
                
                # Clean up camera
                self._cleanup_camera()
                
                # Clear queue
                self._clear_queue()
                
                self.error_handler.logger.info("Frame capture stopped")
                
        except Exception as e:
            self.error_handler.handle_error(e, "camera_stop")
    
    def _clear_queue(self):
        """Clear frame queue"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
    
    def is_running(self) -> bool:
        """Check if capture is running"""
        with self._lock:
            return (self._capture_thread is not None and 
                    self._capture_thread.is_alive() and 
                    not self.stopped.is_set())
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        with self._lock:
            return {
                'frame_count': self.frame_count,
                'queue_size': self.queue.qsize(),
                'consecutive_errors': self.consecutive_errors,
                'is_running': self.is_running(),
                'uptime': time.time() - self.last_frame_time if hasattr(self, 'last_frame_time') else 0
            }
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.stop()
        except:
            pass