import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import threading
import time
from pathlib import Path

class ImageProcessor:
    """
    Image processor untuk CameraAI application
    Mendukung image enhancement, filter applications, dan format conversion
    """
    
    def __init__(self):
        # Thread safety
        self._lock = threading.RLock()
        
        # Processing statistics
        self.processing_times = []
        self.total_images_processed = 0
        
        # Available filters
        self.filters = {
            'none': self._no_filter,
            'blur': self._blur_filter,
            'sharpen': self._sharpen_filter,
            'edge_detect': self._edge_detect_filter,
            'emboss': self._emboss_filter,
            'sepia': self._sepia_filter,
            'grayscale': self._grayscale_filter,
            'invert': self._invert_filter,
            'cartoon': self._cartoon_filter,
            'oil_painting': self._oil_painting_filter,
            'sketch': self._sketch_filter,
            'night_vision': self._night_vision_filter
        }
        
        # Enhancement settings
        self.enhancement_settings = {
            'brightness': 1.0,
            'contrast': 1.0,
            'saturation': 1.0,
            'gamma': 1.0,
            'sharpness': 1.0,
            'noise_reduction': 0.0
        }
    
    def process_image(self, 
                     image: np.ndarray, 
                     filter_name: str = "none",
                     enhancement: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Process image dengan filter dan enhancement
        
        Args:
            image: Input image
            filter_name: Name of filter to apply
            enhancement: Enhancement settings
            
        Returns:
            Processed image
        """
        if image is None or image.size == 0:
            return image
        
        start_time = time.time()
        
        try:
            processed = image.copy()
            
            # Apply enhancement if specified
            if enhancement:
                processed = self._apply_enhancement(processed, enhancement)
            
            # Apply filter
            if filter_name in self.filters:
                processed = self.filters[filter_name](processed)
            else:
                print(f"Warning: Unknown filter '{filter_name}', using 'none'")
                processed = self.filters['none'](processed)
            
            # Update statistics
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            self.total_images_processed += 1
            
            # Keep only last 100 processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return processed
            
        except Exception as e:
            print(f"Image processing error: {e}")
            return image
    
    def _apply_enhancement(self, image: np.ndarray, settings: Dict[str, float]) -> np.ndarray:
        """Apply image enhancement"""
        enhanced = image.copy()
        
        # Brightness and contrast
        if 'brightness' in settings or 'contrast' in settings:
            brightness = settings.get('brightness', 1.0)
            contrast = settings.get('contrast', 1.0)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=(brightness - 1) * 100)
        
        # Saturation
        if 'saturation' in settings and settings['saturation'] != 1.0:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * settings['saturation'], 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Gamma correction
        if 'gamma' in settings and settings['gamma'] != 1.0:
            gamma = settings['gamma']
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        # Sharpness
        if 'sharpness' in settings and settings['sharpness'] != 1.0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * settings['sharpness']
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Noise reduction
        if 'noise_reduction' in settings and settings['noise_reduction'] > 0:
            strength = int(settings['noise_reduction'] * 10)
            enhanced = cv2.bilateralFilter(enhanced, strength, strength * 2, strength / 2)
        
        return enhanced
    
    def _no_filter(self, image: np.ndarray) -> np.ndarray:
        """No filter - return original image"""
        return image
    
    def _blur_filter(self, image: np.ndarray) -> np.ndarray:
        """Blur filter"""
        return cv2.GaussianBlur(image, (15, 15), 0)
    
    def _sharpen_filter(self, image: np.ndarray) -> np.ndarray:
        """Sharpen filter"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _edge_detect_filter(self, image: np.ndarray) -> np.ndarray:
        """Edge detection filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def _emboss_filter(self, image: np.ndarray) -> np.ndarray:
        """Emboss filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
        emboss = cv2.filter2D(gray, -1, kernel) + 128
        return cv2.cvtColor(emboss.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    def _sepia_filter(self, image: np.ndarray) -> np.ndarray:
        """Sepia filter"""
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        sepia = cv2.transform(image, sepia_matrix)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    def _grayscale_filter(self, image: np.ndarray) -> np.ndarray:
        """Grayscale filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def _invert_filter(self, image: np.ndarray) -> np.ndarray:
        """Invert filter"""
        return 255 - image
    
    def _cartoon_filter(self, image: np.ndarray) -> np.ndarray:
        """Cartoon filter"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply median blur
        smooth = cv2.medianBlur(smooth, 7)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        
        # Combine edges with original image
        cartoon = cv2.bitwise_and(image, image, mask=edges)
        
        return cartoon
    
    def _oil_painting_filter(self, image: np.ndarray) -> np.ndarray:
        """Oil painting filter"""
        # Apply bilateral filter for oil painting effect
        oil = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply slight blur for painting texture
        oil = cv2.GaussianBlur(oil, (3, 3), 0)
        
        return oil
    
    def _sketch_filter(self, image: np.ndarray) -> np.ndarray:
        """Sketch filter"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert the image
        inv = 255 - gray
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        
        # Blend the images
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    def _night_vision_filter(self, image: np.ndarray) -> np.ndarray:
        """Night vision filter"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Enhance brightness and contrast
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        
        # Apply green tint for night vision effect
        hsv[:, :, 0] = 60  # Green hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
        
        # Convert back to BGR
        night_vision = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return night_vision
    
    def resize_image(self, image: np.ndarray, width: int, height: int, method: str = "cubic") -> np.ndarray:
        """Resize image"""
        if method == "nearest":
            interpolation = cv2.INTER_NEAREST
        elif method == "linear":
            interpolation = cv2.INTER_LINEAR
        elif method == "cubic":
            interpolation = cv2.INTER_CUBIC
        elif method == "lanczos":
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_CUBIC
        
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    def crop_image(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Crop image"""
        return image[y:y+height, x:x+width]
    
    def rotate_image(self, image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Rotate image"""
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def flip_image(self, image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
        """Flip image"""
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        elif direction == "both":
            return cv2.flip(image, -1)
        else:
            return image
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness"""
        return cv2.convertScaleAbs(image, alpha=1.0, beta=(factor - 1) * 100)
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def get_available_filters(self) -> List[str]:
        """Get list of available filters"""
        return list(self.filters.keys())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_images': 0
            }
        
        return {
            'avg_time': np.mean(self.processing_times),
            'min_time': np.min(self.processing_times),
            'max_time': np.max(self.processing_times),
            'total_images': self.total_images_processed
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.processing_times.clear()
        self.total_images_processed = 0
    
    def save_image(self, image: np.ndarray, filepath: str, quality: int = 95) -> bool:
        """Save image to file"""
        try:
            # Determine format from extension
            ext = Path(filepath).suffix.lower()
            
            if ext in ['.jpg', '.jpeg']:
                # JPEG with quality setting
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success = cv2.imwrite(filepath, image, encode_params)
            elif ext in ['.png']:
                # PNG with compression
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                success = cv2.imwrite(filepath, image, encode_params)
            else:
                # Other formats
                success = cv2.imwrite(filepath, image)
            
            return success
        except Exception as e:
            print(f"Failed to save image: {e}")
            return False
    
    def load_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load image from file"""
        try:
            image = cv2.imread(filepath)
            if image is None:
                print(f"Failed to load image: {filepath}")
                return None
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get image information"""
        if image is None:
            return {}
        
        return {
            'shape': image.shape,
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'dtype': str(image.dtype),
            'size': image.size,
            'memory_size': image.nbytes
        } 