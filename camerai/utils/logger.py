import logging
import logging.handlers
import os
import time
import threading
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json
import traceback
import sys
import numpy as np

class CameraAILogger:
    """
    Structured logger untuk CameraAI application
    Mendukung multiple log levels, performance monitoring, dan error tracking
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True):
        
        # Input validation
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log_level")
        if max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        if backup_count < 0:
            raise ValueError("backup_count must be non-negative")
        
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.performance_logs = []
        self.error_logs = []
        self.start_time = time.time()
        
        # Initialize logger
        self._setup_logger()
        
        # Log startup
        self.info("CameraAI Logger initialized", extra={
            'log_dir': log_dir,
            'log_level': log_level,
            'enable_console': enable_console,
            'enable_file': enable_file
        })
    
    def _setup_logger(self):
        """Setup logger dengan handlers"""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('CameraAI')
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file:
            # Main log file
            main_log_path = os.path.join(self.log_dir, 'camerai.log')
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            main_handler.setLevel(self.log_level)
            main_handler.setFormatter(formatter)
            self.logger.addHandler(main_handler)
            
            # Error log file
            error_log_path = os.path.join(self.log_dir, 'camerai_errors.log')
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)
            
            # Performance log file
            perf_log_path = os.path.join(self.log_dir, 'camerai_performance.log')
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(formatter)
            self.logger.addHandler(perf_handler)
    
    def _log_with_context(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log dengan context information"""
        if extra is None:
            extra = {}
        
        # Add timestamp
        extra['timestamp'] = datetime.now().isoformat()
        extra['uptime'] = time.time() - self.start_time
        
        # Add thread info
        extra['thread_id'] = threading.get_ident()
        extra['thread_name'] = threading.current_thread().name
        
        # Log message
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log_with_context(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, extra)
    
    def log_performance(self, 
                       operation: str, 
                       duration: float, 
                       fps: Optional[float] = None,
                       memory_usage: Optional[float] = None,
                       cpu_usage: Optional[float] = None,
                       extra: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        if extra is None:
            extra = {}
        
        extra.update({
            'operation': operation,
            'duration': duration,
            'fps': fps,
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'log_type': 'performance'
        })
        
        # Store performance log
        with self._lock:
            self.performance_logs.append({
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'duration': duration,
                'fps': fps,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'extra': extra
            })
            
            # Keep only last 1000 performance logs
            if len(self.performance_logs) > 1000:
                self.performance_logs.pop(0)
        
        self.info(f"Performance: {operation} took {duration:.3f}s", extra)
    
    def log_error(self, 
                  error: Exception, 
                  context: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None):
        """Log error dengan stack trace"""
        if extra is None:
            extra = {}
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'context': context,
            'log_type': 'error'
        }
        extra.update(error_info)
        
        # Store error log
        with self._lock:
            self.error_logs.append({
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc(),
                'context': context,
                'extra': extra
            })
            
            # Keep only last 100 error logs
            if len(self.error_logs) > 100:
                self.error_logs.pop(0)
        
        self.error(f"Error: {str(error)}", extra)
    
    def log_module_event(self, 
                        module: str, 
                        event: str, 
                        status: str = "success",
                        details: Optional[Dict[str, Any]] = None):
        """Log module-specific events"""
        if details is None:
            details = {}
        
        extra = {
            'module': module,
            'event': event,
            'status': status,
            'log_type': 'module_event'
        }
        extra.update(details)
        
        message = f"Module {module}: {event} - {status}"
        self.info(message, extra)
    
    def log_gesture_event(self, 
                         gesture: str, 
                         confidence: float, 
                         command: str,
                         extra: Optional[Dict[str, Any]] = None):
        """Log gesture recognition events"""
        if extra is None:
            extra = {}
        
        extra.update({
            'gesture': gesture,
            'confidence': confidence,
            'command': command,
            'log_type': 'gesture'
        })
        
        message = f"Gesture: {gesture} ({confidence:.2f}) -> {command}"
        self.info(message, extra)
    
    def log_camera_event(self, 
                        event: str, 
                        camera_index: int,
                        resolution: Optional[Tuple[int, int]] = None,
                        fps: Optional[float] = None,
                        extra: Optional[Dict[str, Any]] = None):
        """Log camera-related events"""
        if extra is None:
            extra = {}
        
        extra.update({
            'camera_index': camera_index,
            'resolution': resolution,
            'fps': fps,
            'log_type': 'camera'
        })
        
        message = f"Camera {camera_index}: {event}"
        if resolution:
            message += f" ({resolution[0]}x{resolution[1]})"
        if fps:
            message += f" @ {fps:.1f}fps"
        
        self.info(message, extra)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if not self.performance_logs:
                return {
                    'total_operations': 0,
                    'avg_duration': 0.0,
                    'total_errors': len(self.error_logs),
                    'uptime': time.time() - self.start_time
                }
            
            durations = [log['duration'] for log in self.performance_logs]
            fps_values = [log['fps'] for log in self.performance_logs if log['fps'] is not None]
            memory_values = [log['memory_usage'] for log in self.performance_logs if log['memory_usage'] is not None]
            
            return {
                'total_operations': len(self.performance_logs),
                'avg_duration': np.mean(durations) if durations else 0.0,
                'min_duration': np.min(durations) if durations else 0.0,
                'max_duration': np.max(durations) if durations else 0.0,
                'avg_fps': np.mean(fps_values) if fps_values else 0.0,
                'avg_memory': np.mean(memory_values) if memory_values else 0.0,
                'total_errors': len(self.error_logs),
                'uptime': time.time() - self.start_time
            }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self._lock:
            if not self.error_logs:
                return {
                    'total_errors': 0,
                    'error_types': {},
                    'recent_errors': []
                }
            
            # Count error types
            error_types = {}
            for log in self.error_logs:
                error_type = log['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Get recent errors (last 10)
            recent_errors = self.error_logs[-10:] if len(self.error_logs) > 10 else self.error_logs
            
            return {
                'total_errors': len(self.error_logs),
                'error_types': error_types,
                'recent_errors': recent_errors
            }
    
    def export_logs(self, output_file: str, log_type: str = "all"):
        """Export logs to JSON file"""
        with self._lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'log_type': log_type,
                'performance_logs': self.performance_logs if log_type in ["all", "performance"] else [],
                'error_logs': self.error_logs if log_type in ["all", "error"] else [],
                'stats': {
                    'performance': self.get_performance_stats(),
                    'errors': self.get_error_stats()
                }
            }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.info(f"Logs exported to {output_file}")
        except Exception as e:
            self.error(f"Failed to export logs: {e}")
    
    def clear_logs(self, log_type: str = "all"):
        """Clear stored logs"""
        with self._lock:
            if log_type in ["all", "performance"]:
                self.performance_logs.clear()
            if log_type in ["all", "error"]:
                self.error_logs.clear()
        
        self.info(f"Cleared {log_type} logs")
    
    def set_log_level(self, level: str):
        """Set log level"""
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")
        
        self.log_level = getattr(logging, level.upper())
        self.logger.setLevel(self.log_level)
        
        # Update all handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level)
        
        self.info(f"Log level changed to {level}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.info("CameraAI Logger shutting down")


# Global logger instance
_logger_instance = None

def get_logger() -> CameraAILogger:
    """Get global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CameraAILogger()
    return _logger_instance

def setup_logger(**kwargs) -> CameraAILogger:
    """Setup global logger with custom parameters"""
    global _logger_instance
    _logger_instance = CameraAILogger(**kwargs)
    return _logger_instance
