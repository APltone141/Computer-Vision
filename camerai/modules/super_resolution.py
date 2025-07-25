import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import threading
import time

try:
    import torch
    import torch.nn.functional as F
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch/Real-ESRGAN not available")
    print("Falling back to OpenCV upscaling")
    TORCH_AVAILABLE = False

class SuperResolution:
    """
    Super Resolution module untuk meningkatkan resolusi gambar
    Menggunakan Real-ESRGAN dengan fallback ke OpenCV
    """
    
    def __init__(self, scale_factor: float = 2.0, model_name: str = "RealESRGAN_x4plus"):
        # Input validation
        if not 1.0 <= scale_factor <= 4.0:
            raise ValueError("scale_factor must be between 1.0 and 4.0")
        if model_name not in ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-animevideov3"]:
            raise ValueError("Invalid model_name")
        
        self.scale_factor = scale_factor
        self.model_name = model_name
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize models
        self.upscaler = None
        self.model_type = "none"
        
        if TORCH_AVAILABLE:
            self._init_realesrgan()
        else:
            self._init_opencv_fallback()
        
        # Performance tracking
        with self._lock:
            self.processing_times = []
            self.total_frames_processed = 0
            self.last_process_time = 0.0
    
    def _init_realesrgan(self):
        """Initialize Real-ESRGAN models"""
        try:
            if self.model_name == "RealESRGAN_x4plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            elif self.model_name == "RealESRGAN_x4plus_anime_6B":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            elif self.model_name == "realesr-animevideov3":
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            
            # Download model if needed
            model_path = load_file_from_url(file_url[0], model_dir='weights')
            
            # Initialize upscaler
            self.upscaler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
                gpu_id=0
            )
            
            self.model_type = "realesrgan"
            print(f"✅ Real-ESRGAN model loaded: {self.model_name}")
            
        except Exception as e:
            print(f"Real-ESRGAN initialization failed: {e}")
            self._init_opencv_fallback()
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV fallback"""
        self.model_type = "opencv"
        print("✅ Using OpenCV upscaling fallback")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame untuk super resolution
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Upscaled frame
        """
        if frame is None or frame.size == 0:
            return frame
        
        start_time = time.time()
        
        try:
            if self.model_type == "realesrgan" and self.upscaler is not None:
                # Use Real-ESRGAN
                upscaled_frame, _ = self.upscaler.enhance(frame, outscale=self.scale_factor)
            else:
                # Use OpenCV fallback
                upscaled_frame = self._opencv_upscale(frame)
            
            # Update performance stats
            process_time = time.time() - start_time
            with self._lock:
                self.processing_times.append(process_time)
                self.total_frames_processed += 1
                self.last_process_time = process_time
                
                # Keep only last 100 processing times
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
            
            return upscaled_frame
            
        except Exception as e:
            print(f"Super resolution processing error: {e}")
            return frame
    
    def _opencv_upscale(self, frame: np.ndarray) -> np.ndarray:
        """OpenCV upscaling fallback"""
        h, w = frame.shape[:2]
        new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
        
        # Use INTER_CUBIC for better quality
        upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        return sharpened
    
    def process_batch(self, frames: list) -> list:
        """Process multiple frames"""
        if not frames:
            return frames
        
        results = []
        for frame in frames:
            result = self.process(frame)
            results.append(result)
        
        return results
    
    def set_scale_factor(self, scale: float):
        """Set scale factor"""
        if not 1.0 <= scale <= 4.0:
            raise ValueError("Scale factor must be between 1.0 and 4.0")
        
        with self._lock:
            self.scale_factor = scale
    
    def set_model(self, model_name: str):
        """Change model"""
        if model_name not in ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-animevideov3"]:
            raise ValueError("Invalid model name")
        
        with self._lock:
            self.model_name = model_name
            
        # Reinitialize with new model
        if TORCH_AVAILABLE:
            self._init_realesrgan()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if not self.processing_times:
                return {
                    'avg_time': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0,
                    'total_frames': 0,
                    'model_type': self.model_type,
                    'scale_factor': self.scale_factor
                }
            
            return {
                'avg_time': np.mean(self.processing_times),
                'min_time': np.min(self.processing_times),
                'max_time': np.max(self.processing_times),
                'total_frames': self.total_frames_processed,
                'model_type': self.model_type,
                'scale_factor': self.scale_factor,
                'last_process_time': self.last_process_time
            }
    
    def is_available(self) -> bool:
        """Check if super resolution is available"""
        return self.model_type != "none"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'scale_factor': self.scale_factor,
            'torch_available': TORCH_AVAILABLE
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self._lock:
            self.processing_times.clear()
            self.total_frames_processed = 0
            self.last_process_time = 0.0
    
    def optimize_for_realtime(self, target_fps: float = 30.0):
        """Optimize settings for real-time processing"""
        if target_fps <= 0:
            raise ValueError("Target FPS must be positive")
        
        # Calculate target processing time
        target_time = 1.0 / target_fps
        
        with self._lock:
            # Adjust scale factor based on performance
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                if avg_time > target_time:
                    # Reduce scale factor
                    new_scale = max(1.0, self.scale_factor * 0.8)
                    self.scale_factor = new_scale
                    print(f"Reduced scale factor to {new_scale:.2f} for real-time performance")
                elif avg_time < target_time * 0.5:
                    # Increase scale factor
                    new_scale = min(4.0, self.scale_factor * 1.2)
                    self.scale_factor = new_scale
                    print(f"Increased scale factor to {new_scale:.2f} for better quality")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'upscaler') and self.upscaler is not None:
            del self.upscaler
