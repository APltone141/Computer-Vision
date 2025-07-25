import cv2
import numpy as np
import os
import time
import threading
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from pathlib import Path

class VideoRecorder:
    """
    Video recorder untuk CameraAI application
    Mendukung berbagai format video dan quality settings
    """
    
    def __init__(self, 
                 output_dir: str = "recordings",
                 format: str = "mp4",
                 codec: str = "H264",
                 quality: str = "high",
                 fps: float = 30.0,
                 resolution: Tuple[int, int] = (640, 480)):
        
        # Input validation
        if format not in ["mp4", "avi", "mov", "mkv"]:
            raise ValueError("Unsupported format")
        if codec not in ["H264", "H265", "XVID", "MJPG"]:
            raise ValueError("Unsupported codec")
        if quality not in ["low", "medium", "high", "ultra"]:
            raise ValueError("Invalid quality setting")
        if fps <= 0:
            raise ValueError("FPS must be positive")
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("Resolution must be positive")
        
        self.output_dir = output_dir
        self.format = format
        self.codec = codec
        self.quality = quality
        self.fps = fps
        self.resolution = resolution
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Recording state
        with self._lock:
            self.is_recording = False
            self.recording_start_time = None
            self.recording_duration = 0.0
            self.frame_count = 0
            self.current_file = None
            self.writer = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Quality settings
        self.quality_settings = {
            'low': {'crf': 28, 'preset': 'fast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'},
            'ultra': {'crf': 12, 'preset': 'veryslow'}
        }
        
        # Codec mappings
        self.codec_mappings = {
            'H264': cv2.VideoWriter_fourcc(*'H264'),
            'H265': cv2.VideoWriter_fourcc(*'H265'),
            'XVID': cv2.VideoWriter_fourcc(*'XVID'),
            'MJPG': cv2.VideoWriter_fourcc(*'MJPG')
        }
        
        # Recording statistics
        self.total_recordings = 0
        self.total_duration = 0.0
        self.total_frames = 0
    
    def start_recording(self, filename: Optional[str] = None) -> bool:
        """Start video recording"""
        if self.is_recording:
            return False
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camerai_recording_{timestamp}.{self.format}"
            
            # Create full path
            filepath = os.path.join(self.output_dir, filename)
            
            # Get codec
            fourcc = self.codec_mappings.get(self.codec, cv2.VideoWriter_fourcc(*'H264'))
            
            # Create video writer
            self.writer = cv2.VideoWriter(
                filepath,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if not self.writer.isOpened():
                raise RuntimeError("Failed to create video writer")
            
            # Update state
            with self._lock:
                self.is_recording = True
                self.recording_start_time = time.time()
                self.recording_duration = 0.0
                self.frame_count = 0
                self.current_file = filepath
            
            print(f"✅ Recording started: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[str]:
        """Stop video recording"""
        if not self.is_recording:
            return None
        
        try:
            # Stop writer
            if self.writer:
                self.writer.release()
                self.writer = None
            
            # Calculate statistics
            with self._lock:
                if self.recording_start_time:
                    self.recording_duration = time.time() - self.recording_start_time
                
                # Update totals
                self.total_recordings += 1
                self.total_duration += self.recording_duration
                self.total_frames += self.frame_count
                
                # Get filepath
                filepath = self.current_file
                
                # Reset state
                self.is_recording = False
                self.recording_start_time = None
                self.recording_duration = 0.0
                self.frame_count = 0
                self.current_file = None
            
            print(f"✅ Recording stopped: {os.path.basename(filepath)}")
            print(f"   Duration: {self.recording_duration:.2f}s")
            print(f"   Frames: {self.frame_count}")
            print(f"   FPS: {self.frame_count / self.recording_duration:.1f}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to stop recording: {e}")
            return None
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write frame to video"""
        if not self.is_recording or self.writer is None:
            return False
        
        try:
            # Resize frame if needed
            if frame.shape[:2] != self.resolution[::-1]:  # OpenCV uses (height, width)
                frame = cv2.resize(frame, self.resolution)
            
            # Write frame
            self.writer.write(frame)
            
            # Update statistics
            with self._lock:
                self.frame_count += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to write frame: {e}")
            return False
    
    def is_recording_active(self) -> bool:
        """Check if recording is active"""
        with self._lock:
            return self.is_recording
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get current recording statistics"""
        with self._lock:
            current_duration = 0.0
            if self.recording_start_time:
                current_duration = time.time() - self.recording_start_time
            
            return {
                'is_recording': self.is_recording,
                'current_duration': current_duration,
                'frame_count': self.frame_count,
                'current_fps': self.frame_count / current_duration if current_duration > 0 else 0,
                'current_file': self.current_file,
                'total_recordings': self.total_recordings,
                'total_duration': self.total_duration,
                'total_frames': self.total_frames
            }
    
    def set_output_directory(self, directory: str):
        """Set output directory"""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with self._lock:
            self.output_dir = directory
    
    def set_format(self, format: str):
        """Set video format"""
        if format not in ["mp4", "avi", "mov", "mkv"]:
            raise ValueError("Unsupported format")
        
        with self._lock:
            self.format = format
    
    def set_codec(self, codec: str):
        """Set video codec"""
        if codec not in ["H264", "H265", "XVID", "MJPG"]:
            raise ValueError("Unsupported codec")
        
        with self._lock:
            self.codec = codec
    
    def set_quality(self, quality: str):
        """Set video quality"""
        if quality not in ["low", "medium", "high", "ultra"]:
            raise ValueError("Invalid quality setting")
        
        with self._lock:
            self.quality = quality
    
    def set_fps(self, fps: float):
        """Set recording FPS"""
        if fps <= 0:
            raise ValueError("FPS must be positive")
        
        with self._lock:
            self.fps = fps
    
    def set_resolution(self, resolution: Tuple[int, int]):
        """Set recording resolution"""
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("Resolution must be positive")
        
        with self._lock:
            self.resolution = resolution
    
    def get_available_formats(self) -> List[str]:
        """Get list of available formats"""
        return ["mp4", "avi", "mov", "mkv"]
    
    def get_available_codecs(self) -> List[str]:
        """Get list of available codecs"""
        return ["H264", "H265", "XVID", "MJPG"]
    
    def get_available_qualities(self) -> List[str]:
        """Get list of available qualities"""
        return ["low", "medium", "high", "ultra"]
    
    def get_recording_info(self) -> Dict[str, Any]:
        """Get recording configuration info"""
        return {
            'output_dir': self.output_dir,
            'format': self.format,
            'codec': self.codec,
            'quality': self.quality,
            'fps': self.fps,
            'resolution': self.resolution,
            'quality_settings': self.quality_settings[self.quality]
        }
    
    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all recordings in output directory"""
        recordings = []
        
        try:
            for filename in os.listdir(self.output_dir):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    filepath = os.path.join(self.output_dir, filename)
                    stat = os.stat(filepath)
                    
                    recordings.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime),
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
            
            # Sort by creation time (newest first)
            recordings.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            print(f"Failed to list recordings: {e}")
        
        return recordings
    
    def delete_recording(self, filename: str) -> bool:
        """Delete a recording file"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"✅ Deleted recording: {filename}")
                return True
            else:
                print(f"❌ Recording not found: {filename}")
                return False
        except Exception as e:
            print(f"❌ Failed to delete recording: {e}")
            return False
    
    def get_recording_size(self, filename: str) -> Optional[int]:
        """Get size of recording file in bytes"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                return os.path.getsize(filepath)
            return None
        except Exception:
            return None
    
    def cleanup_old_recordings(self, max_age_days: int = 30):
        """Clean up old recordings"""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            deleted_count = 0
            
            for filename in os.listdir(self.output_dir):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    filepath = os.path.join(self.output_dir, filename)
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
            
            if deleted_count > 0:
                print(f"✅ Cleaned up {deleted_count} old recordings")
            
        except Exception as e:
            print(f"❌ Failed to cleanup recordings: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.is_recording:
            self.stop_recording() 