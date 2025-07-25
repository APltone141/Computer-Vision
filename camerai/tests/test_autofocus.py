import unittest
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.autofocus import AutoFocus

class TestAutoFocus(unittest.TestCase):
    """Test cases untuk AutoFocus module"""
    
    def setUp(self):
        """Setup test environment"""
        self.autofocus = AutoFocus()
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test image with face-like region
        self.face_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add a face-like region (simple rectangle)
        cv2.rectangle(self.face_image, (200, 150), (400, 350), (255, 255, 255), -1)
    
    def test_initialization(self):
        """Test AutoFocus initialization"""
        self.assertIsNotNone(self.autofocus)
        self.assertEqual(self.autofocus.padding, 50)
        self.assertEqual(self.autofocus.alpha, 0.4)
        self.assertEqual(self.autofocus.min_crop_ratio, 0.6)
    
    def test_process_none_frame(self):
        """Test processing None frame"""
        result = self.autofocus.process(None)
        self.assertIsNone(result)
    
    def test_process_empty_frame(self):
        """Test processing empty frame"""
        empty_frame = np.array([])
        result = self.autofocus.process(empty_frame)
        self.assertEqual(result.size, 0)
    
    def test_process_valid_frame(self):
        """Test processing valid frame"""
        result = self.autofocus.process(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, self.test_image.dtype)
    
    def test_padding_parameter(self):
        """Test padding parameter"""
        autofocus = AutoFocus(padding=100)
        self.assertEqual(autofocus.padding, 100)
        
        # Test invalid padding
        with self.assertRaises(ValueError):
            AutoFocus(padding=-10)
    
    def test_alpha_parameter(self):
        """Test alpha parameter"""
        autofocus = AutoFocus(alpha=0.8)
        self.assertEqual(autofocus.alpha, 0.8)
        
        # Test invalid alpha
        with self.assertRaises(ValueError):
            AutoFocus(alpha=1.5)
    
    def test_min_crop_ratio_parameter(self):
        """Test min_crop_ratio parameter"""
        autofocus = AutoFocus(min_crop_ratio=0.8)
        self.assertEqual(autofocus.min_crop_ratio, 0.8)
        
        # Test invalid min_crop_ratio
        with self.assertRaises(ValueError):
            AutoFocus(min_crop_ratio=1.5)
    
    def test_apply_crop_method(self):
        """Test _apply_crop method"""
        # Test with valid parameters
        x, y, w, h = 100, 100, 200, 200
        result = self.autofocus._apply_crop(self.test_image, x, y, w, h, 640, 480)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_apply_crop_edge_cases(self):
        """Test _apply_crop with edge cases"""
        # Test with coordinates outside image bounds
        result = self.autofocus._apply_crop(self.test_image, -50, -50, 100, 100, 640, 480)
        self.assertIsNotNone(result)
        
        # Test with very large crop area
        result = self.autofocus._apply_crop(self.test_image, 0, 0, 1000, 1000, 640, 480)
        self.assertIsNotNone(result)
    
    def test_detection_type(self):
        """Test detection type initialization"""
        # Should have a detection type
        self.assertIsNotNone(self.autofocus.detection_type)
        self.assertIn(self.autofocus.detection_type, ["mediapipe", "opencv", "none"])
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism"""
        # Test that fallback works when MediaPipe is not available
        autofocus = AutoFocus()
        self.assertIsNotNone(autofocus.detection_type)
    
    def test_process_with_face_image(self):
        """Test processing image with face-like region"""
        result = self.autofocus.process(self.face_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.face_image.shape)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        try:
            AutoFocus(min_confidence=0.5, padding=30, alpha=0.3, min_crop_ratio=0.5)
        except ValueError:
            self.fail("Valid parameters should not raise ValueError")
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            AutoFocus(min_confidence=1.5)
        
        with self.assertRaises(ValueError):
            AutoFocus(min_confidence=-0.1)
    
    def test_performance(self):
        """Test processing performance"""
        import time
        
        start_time = time.time()
        for _ in range(10):
            self.autofocus.process(self.test_image)
        end_time = time.time()
        
        # Should process 10 frames in reasonable time (less than 1 second)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many frames
        for _ in range(100):
            self.autofocus.process(self.test_image)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)
    
    def test_thread_safety(self):
        """Test thread safety"""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_frames():
            try:
                for _ in range(10):
                    result = self.autofocus.process(self.test_image)
                    results.append(result is not None)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(4):
            thread = threading.Thread(target=process_frames)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Threading errors: {errors}")
        self.assertEqual(len(results), 40)  # 4 threads * 10 frames
        self.assertTrue(all(results))  # All should be successful
    
    def test_consistency(self):
        """Test processing consistency"""
        # Process same image multiple times
        results = []
        for _ in range(5):
            result = self.autofocus.process(self.test_image)
            results.append(result)
        
        # All results should have same shape
        shapes = [r.shape for r in results]
        self.assertTrue(all(s == shapes[0] for s in shapes))
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with corrupted image data
        corrupted_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        corrupted_image[50, 50] = [np.inf, np.inf, np.inf]  # Invalid values
        
        # Should not crash
        try:
            result = self.autofocus.process(corrupted_image)
            self.assertIsNotNone(result)
        except Exception as e:
            # If it crashes, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))


if __name__ == '__main__':
    unittest.main()
