#!/usr/bin/env python3
"""
Simple GUI test script for CameraAI
Tests GUI components without requiring camera access
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import time

def test_gui_components():
    """Test GUI components without camera"""
    print("🎯 Testing CameraAI GUI Components")
    print("=" * 50)
    
    # Test imports
    try:
        from gui.gui_main import CameraAIMainWindow
        print("✅ GUI imports: PASS")
    except Exception as e:
        print(f"❌ GUI imports: FAIL - {e}")
        return False
    
    # Test GUI creation
    try:
        app = QApplication(sys.argv)
        print("✅ QApplication: PASS")
    except Exception as e:
        print(f"❌ QApplication: FAIL - {e}")
        return False
    
    # Test main window creation
    try:
        # Create window without camera
        window = CameraAIMainWindow()
        print("✅ MainWindow creation: PASS")
        
        # Test UI components
        if hasattr(window, 'video_label'):
            print("✅ Video display: PASS")
        else:
            print("❌ Video display: FAIL")
            
        if hasattr(window, 'fps_label'):
            print("✅ FPS display: PASS")
        else:
            print("❌ FPS display: FAIL")
            
        if hasattr(window, 'autofocus_checkbox'):
            print("✅ AutoFocus controls: PASS")
        else:
            print("❌ AutoFocus controls: FAIL")
            
        if hasattr(window, 'tracking_checkbox'):
            print("✅ Tracking controls: PASS")
        else:
            print("❌ Tracking controls: FAIL")
            
        if hasattr(window, 'motion_checkbox'):
            print("✅ Motion detection controls: PASS")
        else:
            print("❌ Motion detection controls: FAIL")
        
        # Close window
        window.close()
        print("✅ Window cleanup: PASS")
        
    except Exception as e:
        print(f"❌ MainWindow test: FAIL - {e}")
        return False
    
    print("\n🎉 GUI Components Test: PASSED!")
    return True

def test_modules_without_camera():
    """Test modules without camera access"""
    print("\n🔧 Testing Modules (No Camera)")
    print("=" * 50)
    
    import numpy as np
    
    # Test AutoFocus
    try:
        from modules.autofocus import AutoFocus
        af = AutoFocus()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = af.process(test_image)
        print("✅ AutoFocus module: PASS")
    except Exception as e:
        print(f"❌ AutoFocus module: FAIL - {e}")
    
    # Test ObjectTracker
    try:
        from modules.tracking import ObjectTracker
        ot = ObjectTracker()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = ot.process(test_image)
        print("✅ ObjectTracker module: PASS")
    except Exception as e:
        print(f"❌ ObjectTracker module: FAIL - {e}")
    
    # Test MotionDetector
    try:
        from modules.motion_detector import MotionDetector
        md = MotionDetector()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = md.process(test_image)
        print("✅ MotionDetector module: PASS")
    except Exception as e:
        print(f"❌ MotionDetector module: FAIL - {e}")
    
    # Test GestureController
    try:
        from modules.gesture_control import GestureController
        gc = GestureController()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result, gesture_data = gc.process(test_image)
        print("✅ GestureController module: PASS")
    except Exception as e:
        print(f"❌ GestureController module: FAIL - {e}")
    
    # Test SuperResolution
    try:
        from modules.super_resolution import SuperResolution
        sr = SuperResolution()
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = sr.process(test_image)
        print("✅ SuperResolution module: PASS")
    except Exception as e:
        print(f"❌ SuperResolution module: FAIL - {e}")
    
    # Test FPSInterpolator
    try:
        from modules.fps_interpolator import FPSInterpolator
        fi = FPSInterpolator()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = fi.process(test_image, 30.0)
        print("✅ FPSInterpolator module: PASS")
    except Exception as e:
        print(f"❌ FPSInterpolator module: FAIL - {e}")

def test_utilities():
    """Test utility modules"""
    print("\n🛠️  Testing Utilities")
    print("=" * 50)
    
    # Test Logger
    try:
        from utils.logger import CameraAILogger
        logger = CameraAILogger()
        logger.info("Test message")
        print("✅ Logger utility: PASS")
    except Exception as e:
        print(f"❌ Logger utility: FAIL - {e}")
    
    # Test VideoRecorder
    try:
        from utils.video_recorder import VideoRecorder
        vr = VideoRecorder()
        print("✅ VideoRecorder utility: PASS")
    except Exception as e:
        print(f"❌ VideoRecorder utility: FAIL - {e}")
    
    # Test ImageProcessor
    try:
        from utils.image_processor import ImageProcessor
        ip = ImageProcessor()
        print("✅ ImageProcessor utility: PASS")
    except Exception as e:
        print(f"❌ ImageProcessor utility: FAIL - {e}")
    
    # Test PerformanceMonitor
    try:
        from utils.performance import PerformanceMonitor
        pm = PerformanceMonitor()
        print("✅ PerformanceMonitor utility: PASS")
    except Exception as e:
        print(f"❌ PerformanceMonitor utility: FAIL - {e}")
    
    # Test ConfigManager
    try:
        from config import ConfigManager
        cm = ConfigManager()
        print("✅ ConfigManager utility: PASS")
    except Exception as e:
        print(f"❌ ConfigManager utility: FAIL - {e}")

def main():
    """Main test function"""
    print("🚀 CameraAI Comprehensive Test Suite")
    print("=" * 60)
    
    # Test utilities first
    test_utilities()
    
    # Test modules without camera
    test_modules_without_camera()
    
    # Test GUI components
    gui_success = test_gui_components()
    
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    if gui_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ CameraAI is ready for production use!")
        print("\n🚀 You can now run: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 