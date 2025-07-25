"""
Error handling utilities for CameraAI application
Provides comprehensive error handling, logging, and recovery mechanisms
"""

import sys
import traceback
import logging
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from functools import wraps
from contextlib import contextmanager
from constants import *

class CameraAIError(Exception):
    """Base exception for CameraAI application"""
    pass

class CameraError(CameraAIError):
    """Camera-related errors"""
    pass

class ProcessingError(CameraAIError):
    """Processing-related errors"""
    pass

class ValidationError(CameraAIError):
    """Validation-related errors"""
    pass

class ConfigurationError(CameraAIError):
    """Configuration-related errors"""
    pass

class PerformanceError(CameraAIError):
    """Performance-related errors"""
    pass

class ErrorHandler:
    """Comprehensive error handling for CameraAI"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.error_history = []
        self.recovery_strategies = {}
        self._lock = threading.RLock()
        
        # Initialize error recovery strategies
        self._init_recovery_strategies()
    
    def _init_recovery_strategies(self):
        """Initialize error recovery strategies"""
        self.recovery_strategies = {
            CameraError: self._recover_camera_error,
            ProcessingError: self._recover_processing_error,
            ValidationError: self._recover_validation_error,
            ConfigurationError: self._recover_configuration_error,
            PerformanceError: self._recover_performance_error
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    retry_count: int = 0,
                    max_retries: int = 3) -> bool:
        """
        Handle error with logging and recovery
        
        Args:
            error: The exception to handle
            context: Context where error occurred
            retry_count: Current retry count
            max_retries: Maximum retries allowed
            
        Returns:
            True if error was handled successfully, False otherwise
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        with self._lock:
            # Update error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Add to error history
            self.error_history.append({
                'timestamp': time.time(),
                'error_type': error_type,
                'error_message': error_message,
                'context': context,
                'retry_count': retry_count,
                'traceback': traceback.format_exc()
            })
            
            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history.pop(0)
        
        # Log error
        self.logger.error(
            f"Error in {context}: {error_type} - {error_message}",
            extra={
                'error_type': error_type,
                'context': context,
                'retry_count': retry_count,
                'traceback': traceback.format_exc()
            }
        )
        
        # Attempt recovery
        if retry_count < max_retries:
            return self._attempt_recovery(error, context, retry_count, max_retries)
        
        return False
    
    def _attempt_recovery(self, 
                         error: Exception, 
                         context: str, 
                         retry_count: int, 
                         max_retries: int) -> bool:
        """Attempt to recover from error"""
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_type]
                if recovery_func(error, context, retry_count, max_retries):
                    self.logger.info(f"Recovery successful for {error_type.__name__}")
                    return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
        
        return False
    
    def _recover_camera_error(self, error: CameraError, context: str, retry_count: int, max_retries: int) -> bool:
        """Recover from camera errors"""
        if "initialization" in str(error).lower():
            # Wait and retry camera initialization
            time.sleep(ERROR_RETRY_DELAY * (retry_count + 1))
            return True
        elif "not responding" in str(error).lower():
            # Try to restart camera
            time.sleep(ERROR_RETRY_DELAY * (retry_count + 1))
            return True
        return False
    
    def _recover_processing_error(self, error: ProcessingError, context: str, retry_count: int, max_retries: int) -> bool:
        """Recover from processing errors"""
        if "memory" in str(error).lower():
            # Force garbage collection and retry
            import gc
            gc.collect()
            time.sleep(ERROR_RETRY_DELAY)
            return True
        elif "timeout" in str(error).lower():
            # Increase timeout and retry
            time.sleep(ERROR_RETRY_DELAY * (retry_count + 1))
            return True
        return False
    
    def _recover_validation_error(self, error: ValidationError, context: str, retry_count: int, max_retries: int) -> bool:
        """Recover from validation errors"""
        # Validation errors typically don't need recovery
        # Just log and continue
        return False
    
    def _recover_configuration_error(self, error: ConfigurationError, context: str, retry_count: int, max_retries: int) -> bool:
        """Recover from configuration errors"""
        if "file not found" in str(error).lower():
            # Try to create default configuration
            return True
        return False
    
    def _recover_performance_error(self, error: PerformanceError, context: str, retry_count: int, max_retries: int) -> bool:
        """Recover from performance errors"""
        if "memory" in str(error).lower():
            # Force garbage collection
            import gc
            gc.collect()
            return True
        elif "cpu" in str(error).lower():
            # Reduce processing load
            time.sleep(ERROR_RETRY_DELAY)
            return True
        return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self._lock:
            return {
                'total_errors': len(self.error_history),
                'error_counts': self.error_counts.copy(),
                'recent_errors': self.error_history[-10:] if self.error_history else []
            }
    
    def clear_error_history(self):
        """Clear error history"""
        with self._lock:
            self.error_history.clear()
            self.error_counts.clear()

class ErrorDecorator:
    """Decorator for automatic error handling"""
    
    def __init__(self, error_handler: ErrorHandler, context: str = ""):
        self.error_handler = error_handler
        self.context = context
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error_handler.handle_error(e, self.context or func.__name__)
                raise
        return wrapper

@contextmanager
def error_context(error_handler: ErrorHandler, context: str = ""):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        error_handler.handle_error(e, context)
        raise

class PerformanceMonitor:
    """Monitor performance and detect issues"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.performance_thresholds = {
            'max_memory_mb': PERFORMANCE_MAX_MEMORY_USAGE,
            'max_cpu_percent': PERFORMANCE_MAX_CPU_USAGE,
            'min_fps': PERFORMANCE_MIN_FPS,
            'max_processing_time': PROCESSING_TIMEOUT
        }
        self._lock = threading.RLock()
    
    def check_performance(self, 
                         memory_usage: float = None,
                         cpu_usage: float = None,
                         fps: float = None,
                         processing_time: float = None) -> List[str]:
        """Check performance metrics and return warnings"""
        warnings = []
        
        if memory_usage and memory_usage > self.performance_thresholds['max_memory_mb']:
            warnings.append(f"Memory usage too high: {memory_usage:.1f}MB")
        
        if cpu_usage and cpu_usage > self.performance_thresholds['max_cpu_percent']:
            warnings.append(f"CPU usage too high: {cpu_usage:.1f}%")
        
        if fps and fps < self.performance_thresholds['min_fps']:
            warnings.append(f"FPS too low: {fps:.1f}")
        
        if processing_time and processing_time > self.performance_thresholds['max_processing_time']:
            warnings.append(f"Processing time too high: {processing_time:.3f}s")
        
        return warnings
    
    def handle_performance_warnings(self, warnings: List[str]):
        """Handle performance warnings"""
        for warning in warnings:
            self.error_handler.logger.warning(warning)
            
            # Create performance error if severe
            if "Memory usage too high" in warning or "CPU usage too high" in warning:
                error = PerformanceError(warning)
                self.error_handler.handle_error(error, "performance_monitor")

class ThreadSafetyManager:
    """Manage thread safety and prevent race conditions"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.locks = {}
        self._lock = threading.RLock()
    
    def get_lock(self, name: str) -> threading.RLock:
        """Get or create a lock for a specific resource"""
        with self._lock:
            if name not in self.locks:
                self.locks[name] = threading.RLock()
            return self.locks[name]
    
    @contextmanager
    def safe_operation(self, lock_name: str, timeout: float = THREAD_LOCK_TIMEOUT):
        """Context manager for thread-safe operations"""
        lock = self.get_lock(lock_name)
        
        try:
            if lock.acquire(timeout=timeout):
                try:
                    yield
                finally:
                    lock.release()
            else:
                raise TimeoutError(f"Lock acquisition timeout for {lock_name}")
        except Exception as e:
            self.error_handler.handle_error(e, f"thread_safety_{lock_name}")
            raise

class ErrorRecoveryManager:
    """Manage error recovery strategies"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.recovery_attempts = {}
        self._lock = threading.RLock()
    
    def should_retry(self, error: Exception, context: str) -> bool:
        """Determine if retry should be attempted"""
        error_key = f"{type(error).__name__}_{context}"
        
        with self._lock:
            attempts = self.recovery_attempts.get(error_key, 0)
            if attempts < ERROR_MAX_CONSECUTIVE_ERRORS:
                self.recovery_attempts[error_key] = attempts + 1
                return True
            else:
                # Reset counter after some time
                if time.time() - self.recovery_attempts.get(f"{error_key}_time", 0) > 60:
                    self.recovery_attempts[error_key] = 0
                    return True
                return False
    
    def reset_recovery_attempts(self, context: str):
        """Reset recovery attempts for a context"""
        with self._lock:
            keys_to_remove = [k for k in self.recovery_attempts.keys() if context in k]
            for key in keys_to_remove:
                del self.recovery_attempts[key]

# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def set_global_error_handler(error_handler: ErrorHandler):
    """Set global error handler instance"""
    global _global_error_handler
    _global_error_handler = error_handler 