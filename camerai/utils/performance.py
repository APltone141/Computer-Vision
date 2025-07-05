import time
import psutil
import threading
from collections import deque
import numpy as np


class PerformanceMonitor:
    """Monitor system performance untuk aplikasi real-time"""
    
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.reset_stats()
        
        # System monitoring
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        
    def reset_stats(self):
        """Reset semua statistik"""
        self.frame_times = deque(maxlen=self.history_size)
        self.fps_history = deque(maxlen=self.history_size)
        self.cpu_usage = deque(maxlen=self.history_size)
        self.memory_usage = deque(maxlen=self.history_size)
        
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def start_monitoring(self):
        """Start background system monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Background loop untuk monitoring sistem"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_usage.append(memory_mb)
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                print(f"Monitor error: {e}")
                break
    
    def frame_start(self):
        """Mark start of frame processing"""
        self.frame_start_time = time.time()
    
    def frame_end(self):
        """Mark end of frame processing and update stats"""
        current_time = time.time()
        
        # Calculate frame time
        if hasattr(self, 'frame_start_time'):
            frame_time = current_time - self.frame_start_time
            self.frame_times.append(frame_time * 1000)  # Convert to ms
        
        # Calculate FPS
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            fps = 1.0 / time_diff
            self.fps_history.append(fps)
        
        self.last_frame_time = current_time
        self.frame_count += 1
    
    def get_stats(self):
        """Get current performance statistics"""
        stats = {
            'frame_count': self.frame_count,
            'uptime': time.time() - self.start_time,
            'current_fps': self.fps_history[-1] if self.fps_history else 0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'min_fps': np.min(self.fps_history) if self.fps_history else 0,
            'max_fps': np.max(self.fps_history) if self.fps_history else 0,
            'current_frame_time': self.frame_times[-1] if self.frame_times else 0,
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'min_frame_time': np.min(self.frame_times) if self.frame_times else 0,
            'max_frame_time': np.max(self.frame_times) if self.frame_times else 0,
            'current_cpu': self.cpu_usage[-1] if self.cpu_usage else 0,
            'avg_cpu': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'current_memory': self.memory_usage[-1] if self.memory_usage else 0,
            'avg_memory': np.mean(self.memory_usage) if self.memory_usage else 0,
        }
        
        return stats
    
    def get_formatted_stats(self):
        """Get formatted statistics string"""
        stats = self.get_stats()
        
        return f"""
Performance Statistics:
======================
Uptime: {stats['uptime']:.1f}s
Frames: {stats['frame_count']}

FPS:
  Current: {stats['current_fps']:.1f}
  Average: {stats['avg_fps']:.1f}
  Min/Max: {stats['min_fps']:.1f}/{stats['max_fps']:.1f}

Frame Time:
  Current: {stats['current_frame_time']:.1f}ms
  Average: {stats['avg_frame_time']:.1f}ms
  Min/Max: {stats['min_frame_time']:.1f}/{stats['max_frame_time']:.1f}ms

System:
  CPU: {stats['current_cpu']:.1f}% (avg: {stats['avg_cpu']:.1f}%)
  Memory: {stats['current_memory']:.1f}MB (avg: {stats['avg_memory']:.1f}MB)
        """.strip()
    
    def print_stats(self):
        """Print current statistics"""
        print(self.get_formatted_stats())
    
    def is_performance_good(self, min_fps=15, max_frame_time=100):
        """Check if performance is acceptable"""
        stats = self.get_stats()
        

        return (
            stats['current_fps'] >= min_fps and
            stats['current_frame_time'] <= max_frame_time
        )