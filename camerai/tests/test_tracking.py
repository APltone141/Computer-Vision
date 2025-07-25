import unittest
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.tracking import ObjectTracker

class TestObjectTracker(unittest.TestCase):
    """Test cases untuk ObjectTracker module"""
    
    def setUp(self):
        """Setup test environment"""
        self.tracker = ObjectTracker()
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test image with face-like region
        self.face_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add a face-like region (simple rectangle)
        cv2.rectangle(self.face_image, (200, 150), (400, 350), (255, 255, 255), -1)
    
    def test_initialization(self):
        """Test ObjectTracker initialization"""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.tracking_mode, "face")
        self.assertEqual(self.tracker.min_confidence, 0.6)
        self.assertEqual(self.tracker.smoothing_factor, 0.3)
    
    def test_process_none_frame(self):
        """Test processing None frame"""
        result = self.tracker.process(None)
        self.assertIsNone(result)
    
    def test_process_empty_frame(self):
        """Test processing empty frame"""
        empty_frame = np.array([])
        result = self.tracker.process(empty_frame)
        self.assertEqual(result.size, 0)
    
    def test_process_valid_frame(self):
        """Test processing valid frame"""
        result = self.tracker.process(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, self.test_image.dtype)
    
    def test_tracking_mode_parameter(self):
        """Test tracking mode parameter"""
        tracker = ObjectTracker(tracking_mode="hand")
        self.assertEqual(tracker.tracking_mode, "hand")
        
        # Test invalid tracking mode
        with self.assertRaises(ValueError):
            ObjectTracker(tracking_mode="invalid")
    
    def test_min_confidence_parameter(self):
        """Test min_confidence parameter"""
        tracker = ObjectTracker(min_confidence=0.8)
        self.assertEqual(tracker.min_confidence, 0.8)
        
        # Test invalid min_confidence
        with self.assertRaises(ValueError):
            ObjectTracker(min_confidence=1.5)
        
        with self.assertRaises(ValueError):
            ObjectTracker(min_confidence=-0.1)
    
    def test_smoothing_factor_parameter(self):
        """Test smoothing_factor parameter"""
        tracker = ObjectTracker(smoothing_factor=0.8)
        self.assertEqual(tracker.smoothing_factor, 0.8)
        
        # Test invalid smoothing_factor
        with self.assertRaises(ValueError):
            ObjectTracker(smoothing_factor=1.5)
        
        with self.assertRaises(ValueError):
            ObjectTracker(smoothing_factor=-0.1)
    
    def test_detection_type(self):
        """Test detection type initialization"""
        # Should have a detection type
        self.assertIsNotNone(self.tracker.detection_type)
        self.assertIn(self.tracker.detection_type, 
                     ["mediapipe_face", "mediapipe_hand", "opencv_face", "none"])
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism"""
        # Test that fallback works when MediaPipe is not available
        tracker = ObjectTracker()
        self.assertIsNotNone(tracker.detection_type)
    
    def test_process_with_face_image(self):
        """Test processing image with face-like region"""
        result = self.tracker.process(self.face_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.face_image.shape)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        try:
            ObjectTracker(tracking_mode="face", min_confidence=0.5, smoothing_factor=0.3)
        except ValueError:
            self.fail("Valid parameters should not raise ValueError")
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            ObjectTracker(tracking_mode="invalid")
        
        with self.assertRaises(ValueError):
            ObjectTracker(min_confidence=1.5)
    
    def test_set_zoom_factor(self):
        """Test set_zoom_factor method"""
        # Test valid zoom factor
        self.tracker.set_zoom_factor(1.5)
        self.assertEqual(self.tracker.zoom_factor, 1.5)
        
        # Test invalid zoom factor
        with self.assertRaises(ValueError):
            self.tracker.set_zoom_factor(0.1)  # Too small
        
        with self.assertRaises(ValueError):
            self.tracker.set_zoom_factor(5.0)  # Too large
    
    def test_set_smoothing_factor(self):
        """Test set_smoothing_factor method"""
        # Test valid smoothing factor
        self.tracker.set_smoothing_factor(0.8)
        self.assertEqual(self.tracker.smoothing_factor, 0.8)
        
        # Test invalid smoothing factor
        with self.assertRaises(ValueError):
            self.tracker.set_smoothing_factor(1.5)
    
    def test_set_max_offset(self):
        """Test set_max_offset method"""
        # Test valid max offset
        self.tracker.set_max_offset(0.4)
        self.assertEqual(self.tracker.max_offset, 0.4)
        
        # Test invalid max offset
        with self.assertRaises(ValueError):
            self.tracker.set_max_offset(0.05)  # Too small
        
        with self.assertRaises(ValueError):
            self.tracker.set_max_offset(0.8)  # Too large
    
    def test_reset_tracking(self):
        """Test reset_tracking method"""
        # Set some tracking state
        self.tracker.target_center = (0.5, 0.5)
        self.tracker.is_tracking = True
        
        # Reset tracking
        self.tracker.reset_tracking()
        
        # Check reset
        self.assertIsNone(self.tracker.target_center)
        self.assertFalse(self.tracker.is_tracking)
    
    def test_get_tracking_status(self):
        """Test get_tracking_status method"""
        status = self.tracker.get_tracking_status()
        
        # Check status structure
        self.assertIsInstance(status, dict)
        self.assertIn('is_tracking', status)
        self.assertIn('target_center', status)
        self.assertIn('frame_offset', status)
        self.assertIn('zoom_factor', status)
        self.assertIn('detection_type', status)
    
    def test_performance(self):
        """Test processing performance"""
        import time
        
        start_time = time.time()
        for _ in range(10):
            self.tracker.process(self.test_image)
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
            self.tracker.process(self.test_image)
        
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
                    result = self.tracker.process(self.test_image)
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
            result = self.tracker.process(self.test_image)
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
            result = self.tracker.process(corrupted_image)
            self.assertIsNotNone(result)
        except Exception as e:
            # If it crashes, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_different_tracking_modes(self):
        """Test different tracking modes"""
        # Test face tracking
        face_tracker = ObjectTracker(tracking_mode="face")
        result = face_tracker.process(self.test_image)
        self.assertIsNotNone(result)
        
        # Test hand tracking
        hand_tracker = ObjectTracker(tracking_mode="hand")
        result = hand_tracker.process(self.test_image)
        self.assertIsNotNone(result)
    
    def test_virtual_camera_transformation(self):
        """Test virtual camera transformation"""
        # Test that transformation doesn't crash
        result = self.tracker._apply_virtual_camera(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_tracking_info_drawing(self):
        """Test tracking info drawing"""
        # Test that drawing doesn't crash
        result = self.tracker._draw_tracking_info(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)


if __name__ == '__main__':
    unittest.main()
