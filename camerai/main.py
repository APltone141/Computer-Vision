#!/usr/bin/env python3
"""
CameraAI - AI Computer Vision Toolkit
Main entry point for the application
"""

import sys
import argparse
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.gui_main import main as gui_main
from camera_handler.webcam import FrameGrabber
from modules.autofocus import AutoFocus
from modules.tracking import ObjectTracker
from modules.motion_detector import MotionDetector


def run_cli_mode(args):
    """Run CameraAI in CLI mode (without GUI)"""
    print("🎥 CameraAI - CLI Mode")
    print("Press ESC to exit")
    
    # Initialize modules
    grabber = FrameGrabber(src=args.camera).start()
    
    modules = {}
    if args.autofocus:
        modules['autofocus'] = AutoFocus(min_confidence=0.6, padding=50)
        print("✅ AutoFocus enabled")
    
    if args.tracking:
        modules['tracker'] = ObjectTracker(tracking_mode=args.tracking_mode)
        print(f"✅ Tracking enabled (mode: {args.tracking_mode})")
    
    if args.motion:
        modules['motion'] = MotionDetector(sensitivity=args.motion_sensitivity)
        print("✅ Motion detection enabled")
    
    try:
        while True:
            frame = grabber.read()
            if frame is None:
                continue
            
            # Process frame through enabled modules
            processed_frame = frame.copy()
            
            if 'autofocus' in modules:
                processed_frame = modules['autofocus'].process(processed_frame)
            
            if 'tracker' in modules:
                processed_frame = modules['tracker'].process(processed_frame)
            
            if 'motion' in modules:
                processed_frame = modules['motion'].process(processed_frame)
            
            # Display result
            cv2.imshow("CameraAI", processed_frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping CameraAI...")
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


def run_gui_mode():
    """Run CameraAI in GUI mode"""
    print("🎥 CameraAI - GUI Mode")
    gui_main()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CameraAI - AI Computer Vision Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with GUI
  %(prog)s --cli                    # Run in CLI mode
  %(prog)s --cli --autofocus        # CLI with autofocus
  %(prog)s --cli --tracking --motion # CLI with tracking and motion
  %(prog)s --camera 1               # Use camera index 1
        """
    )
    
    parser.add_argument(
        '--cli', 
        action='store_true',
        help='Run in CLI mode (no GUI)'
    )
    
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--autofocus', 
        action='store_true',
        help='Enable AutoFocus in CLI mode'
    )
    
    parser.add_argument(
        '--tracking', 
        action='store_true',
        help='Enable object tracking in CLI mode'
    )
    
    parser.add_argument(
        '--tracking-mode', 
        choices=['face', 'hand'],
        default='face',
        help='Tracking mode (default: face)'
    )
    
    parser.add_argument(
        '--motion', 
        action='store_true',
        help='Enable motion detection in CLI mode'
    )
    
    parser.add_argument(
        '--motion-sensitivity', 
        type=int, 
        default=50,
        help='Motion detection sensitivity (default: 50)'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='CameraAI v1.0'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 50)
    print("🎥 CameraAI - AI Computer Vision Toolkit")
    print("   Transform your webcam with AI!")
    print("=" * 50)
    
    # Run appropriate mode
    if args.cli:
        run_cli_mode(args)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()