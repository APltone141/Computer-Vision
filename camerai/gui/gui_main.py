import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QCheckBox, QSlider, QPushButton,
                           QGroupBox, QGridLayout, QComboBox, QSpinBox, QFrame,
                           QTabWidget, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt as QtCoreQt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
# Import Qt constants explicitly to avoid linter issues
from PyQt5.QtCore import Qt
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

# Import modules
from camera_handler.webcam import FrameGrabber
from modules.autofocus import AutoFocus
from modules.tracking import ObjectTracker
from modules.motion_detector import MotionDetector


class VideoProcessor(QThread):
    """Thread untuk memproses video secara real-time dengan optimasi"""
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.frame_grabber = None
        self.running = False
        
        # Initialize modules
        self.autofocus = AutoFocus()
        self.tracker = ObjectTracker()
        self.motion_detector = MotionDetector()
        
        # Module states
        self.autofocus_enabled = False
        self.tracking_enabled = False
        self.motion_enabled = False
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._processing_lock = threading.Lock()
        self._frame_buffer = None
        self._last_processed_frame = None
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Error handling
        self.consecutive_errors = 0
        self.max_errors = 5
        
    def start_capture(self):
        """Start video capture with error handling"""
        try:
            self.frame_grabber = FrameGrabber(src=self.camera_index).start()
            self.running = True
            self.start()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Error starting camera: {e}")
            return False
    
    def stop_capture(self):
        """Stop video capture with proper cleanup"""
        self.running = False
        if self.frame_grabber:
            self.frame_grabber.stop()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.quit()
        self.wait()
    
    def run(self):
        """Main processing loop with optimizations"""
        while self.running:
            if not self.frame_grabber:
                continue
                
            try:
                frame = self.frame_grabber.read()
                if frame is None:
                    continue
                
                # Process frame through enabled modules with parallel processing
                processed_frame = self.process_frame_optimized(frame)
                
                # Calculate FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_updated.emit(self.current_fps)
                    self.fps_counter = 0
                    self.fps_time = current_time
                
                # Emit processed frame
                self.frame_ready.emit(processed_frame)
                
                # Emit stats
                self.emit_stats()
                
                # Reset error counter on success
                self.consecutive_errors = 0
                
            except Exception as e:
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_errors:
                    self.error_occurred.emit(f"Too many consecutive errors: {e}")
                    break
                print(f"Processing error: {e}")
                continue
    
    def process_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with parallel processing and memory optimization"""
        if frame is None or frame.size == 0:
            return frame
        
        # Check if processing is needed
        if not any([self.autofocus_enabled, self.tracking_enabled, self.motion_enabled]):
            return frame  # No processing needed
        
        # Use frame view instead of copy when possible
        processed = frame
        
        # Parallel processing for independent modules
        futures = []
        
        if self.autofocus_enabled:
            futures.append(self.executor.submit(self.autofocus.process, frame.copy()))
        
        if self.tracking_enabled:
            futures.append(self.executor.submit(self.tracker.process, frame.copy()))
        
        if self.motion_enabled:
            futures.append(self.executor.submit(self.motion_detector.process, frame.copy()))
        
        # Wait for all processing to complete
        if futures:
            try:
                results = [f.result(timeout=1.0) for f in futures]
                # Merge results (for now, use the last result)
                processed = results[-1] if results else frame
            except Exception as e:
                print(f"Parallel processing error: {e}")
                processed = frame
        
        return processed
    
    def emit_stats(self):
        """Emit comprehensive statistics"""
        stats = {
            'fps': self.current_fps,
            'autofocus_active': self.autofocus_enabled,
            'tracking_active': self.tracking_enabled,
            'motion_active': self.motion_enabled,
            'motion_stats': self.motion_detector.get_motion_stats() if self.motion_enabled else None,
            'tracking_stats': self.tracker.get_tracking_status() if self.tracking_enabled else None,
            'queue_size': self.frame_grabber.get_queue_size() if self.frame_grabber else 0
        }
        self.stats_updated.emit(stats)


class CameraAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CameraAI - AI Computer Vision Toolkit")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        self.video_processor.frame_ready.connect(self.update_frame)
        self.video_processor.fps_updated.connect(self.update_fps)
        self.video_processor.stats_updated.connect(self.update_stats)
        self.video_processor.error_occurred.connect(self.handle_error)
        
        # Performance monitoring
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Setup UI
        self.setup_ui()
        
        # Start video capture
        self.start_camera()

        # Notifikasi fallback AI
        self.check_ai_fallback()
    
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        self.setup_video_panel(main_layout)
        
        # Right panel - Controls
        self.setup_control_panel(main_layout)
    
    def setup_video_panel(self, main_layout):
        """Setup video display panel"""
        video_group = QGroupBox("Video Feed")
        video_layout = QVBoxLayout(video_group)
        
        # Video display
        self.video_label = QLabel("Camera Starting...")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCoreQt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        video_layout.addWidget(self.video_label)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Initializing")
        self.fps_label = QLabel("FPS: 0.0")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fps_label)
        video_layout.addLayout(status_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        video_layout.addLayout(button_layout)
        
        main_layout.addWidget(video_group, 2)
    
    def setup_control_panel(self, main_layout):
        """Setup control panel"""
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Create tab widget for organized controls
        tab_widget = QTabWidget()
        
        # Module controls tab
        self.setup_module_controls(tab_widget)
        
        # Settings tab
        self.setup_settings_tab(tab_widget)
        
        # Statistics tab
        self.setup_stats_tab(tab_widget)
        
        control_layout.addWidget(tab_widget)
        main_layout.addWidget(control_group, 1)
    
    def setup_module_controls(self, tab_widget):
        """Setup module control tab"""
        module_widget = QWidget()
        module_layout = QVBoxLayout(module_widget)
        
        # AutoFocus controls
        autofocus_group = QGroupBox("AutoFocus")
        autofocus_layout = QGridLayout(autofocus_group)
        
        self.autofocus_checkbox = QCheckBox("Enable AutoFocus")
        self.autofocus_checkbox.toggled.connect(self.toggle_autofocus)
        autofocus_layout.addWidget(self.autofocus_checkbox, 0, 0, 1, 2)
        
        # AutoFocus parameters
        autofocus_layout.addWidget(QLabel("Padding:"), 1, 0)
        self.padding_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.padding_slider.setRange(10, 100)
        self.padding_slider.setValue(50)
        self.padding_slider.valueChanged.connect(self.update_autofocus_params)
        autofocus_layout.addWidget(self.padding_slider, 1, 1)
        
        self.padding_label = QLabel("50")
        autofocus_layout.addWidget(self.padding_label, 1, 2)
        
        autofocus_layout.addWidget(QLabel("Alpha:"), 2, 0)
        self.alpha_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 90)
        self.alpha_slider.setValue(40)
        self.alpha_slider.valueChanged.connect(self.update_autofocus_params)
        autofocus_layout.addWidget(self.alpha_slider, 2, 1)
        
        self.alpha_label = QLabel("0.40")
        autofocus_layout.addWidget(self.alpha_label, 2, 2)
        
        autofocus_layout.addWidget(QLabel("Min Crop:"), 3, 0)
        self.min_crop_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.min_crop_slider.setRange(30, 90)
        self.min_crop_slider.setValue(60)
        self.min_crop_slider.valueChanged.connect(self.update_autofocus_params)
        autofocus_layout.addWidget(self.min_crop_slider, 3, 1)
        
        self.min_crop_label = QLabel("0.60")
        autofocus_layout.addWidget(self.min_crop_label, 3, 2)
        
        module_layout.addWidget(autofocus_group)
        
        # Tracking controls
        tracking_group = QGroupBox("Object Tracking")
        tracking_layout = QGridLayout(tracking_group)
        
        self.tracking_checkbox = QCheckBox("Enable Tracking")
        self.tracking_checkbox.toggled.connect(self.toggle_tracking)
        tracking_layout.addWidget(self.tracking_checkbox, 0, 0, 1, 2)
        
        tracking_layout.addWidget(QLabel("Mode:"), 1, 0)
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems(["face", "hand"])
        self.tracking_mode_combo.currentTextChanged.connect(self.update_tracking_mode)
        tracking_layout.addWidget(self.tracking_mode_combo, 1, 1)
        
        tracking_layout.addWidget(QLabel("Smoothing:"), 2, 0)
        self.track_smooth_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.track_smooth_slider.setRange(10, 90)
        self.track_smooth_slider.setValue(30)
        self.track_smooth_slider.valueChanged.connect(self.update_tracking_params)
        tracking_layout.addWidget(self.track_smooth_slider, 2, 1)
        
        self.track_smooth_label = QLabel("0.30")
        tracking_layout.addWidget(self.track_smooth_label, 2, 2)
        
        tracking_layout.addWidget(QLabel("Zoom:"), 3, 0)
        self.zoom_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_tracking_params)
        tracking_layout.addWidget(self.zoom_slider, 3, 1)
        
        self.zoom_label = QLabel("1.00")
        tracking_layout.addWidget(self.zoom_label, 3, 2)
        
        module_layout.addWidget(tracking_group)
        
        # Motion detection controls
        motion_group = QGroupBox("Motion Detection")
        motion_layout = QGridLayout(motion_group)
        
        self.motion_checkbox = QCheckBox("Enable Motion Detection")
        self.motion_checkbox.toggled.connect(self.toggle_motion)
        motion_layout.addWidget(self.motion_checkbox, 0, 0, 1, 2)
        
        motion_layout.addWidget(QLabel("Sensitivity:"), 1, 0)
        self.sensitivity_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(10, 200)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.valueChanged.connect(self.update_motion_params)
        motion_layout.addWidget(self.sensitivity_slider, 1, 1)
        
        self.sensitivity_label = QLabel("50")
        motion_layout.addWidget(self.sensitivity_label, 1, 2)
        
        motion_layout.addWidget(QLabel("Min Area:"), 2, 0)
        self.min_area_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.min_area_slider.setRange(100, 5000)
        self.min_area_slider.setValue(500)
        self.min_area_slider.valueChanged.connect(self.update_motion_params)
        motion_layout.addWidget(self.min_area_slider, 2, 1)
        
        self.min_area_label = QLabel("500")
        motion_layout.addWidget(self.min_area_label, 2, 2)
        
        module_layout.addWidget(motion_group)
        
        tab_widget.addTab(module_widget, "Modules")
    
    def setup_settings_tab(self, tab_widget):
        """Setup settings tab"""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QGridLayout(camera_group)
        
        camera_layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        camera_layout.addWidget(self.camera_combo, 0, 1)
        
        settings_layout.addWidget(camera_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QGridLayout(perf_group)
        
        perf_layout.addWidget(QLabel("Queue Size:"), 0, 0)
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(16, 128)
        self.queue_size_spin.setValue(64)
        perf_layout.addWidget(self.queue_size_spin, 0, 1)
        
        settings_layout.addWidget(perf_group)
        
        tab_widget.addTab(settings_widget, "Settings")
    
    def setup_stats_tab(self, tab_widget):
        """Setup statistics tab"""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # Detailed FPS
        self.fps_detailed_label = QLabel("FPS: 0.0")
        self.fps_detailed_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        stats_layout.addWidget(self.fps_detailed_label)
        
        # Statistics text
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        tab_widget.addTab(stats_widget, "Statistics")
    
    def start_camera(self):
        """Start video capture"""
        if self.video_processor.start_capture():
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Status: Running")
        else:
            self.status_label.setText("Status: Failed to start")
    
    def stop_camera(self):
        """Stop video capture"""
        self.video_processor.stop_capture()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        self.video_label.setText("Camera Stopped")
    
    def change_camera(self, index):
        """Change camera index"""
        self.video_processor.stop_capture()
        self.video_processor.camera_index = index
        self.start_camera()
    
    def update_frame(self, frame):
        """Update video display with optimization"""
        if frame is None or frame.size == 0:
            return
        
        h, w = frame.shape[:2]
        bytes_per_line = 3 * w
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), QtCoreQt.AspectRatioMode.KeepAspectRatio)
        
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update frame count
        self.frame_count += 1
    
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.fps_detailed_label.setText(f"FPS: {fps:.1f}")
    
    def update_stats(self, stats):
        """Update statistics display"""
        stats_text = f"FPS: {stats['fps']:.1f}\n"
        stats_text += f"AutoFocus: {'Active' if stats['autofocus_active'] else 'Inactive'}\n"
        stats_text += f"Tracking: {'Active' if stats['tracking_active'] else 'Inactive'}\n"
        stats_text += f"Motion: {'Active' if stats['motion_active'] else 'Inactive'}\n"
        stats_text += f"Queue Size: {stats['queue_size']}\n"
        
        if stats['motion_stats']:
            motion_stats = stats['motion_stats']
            stats_text += f"Motion Detected: {'Yes' if motion_stats['detected'] else 'No'}\n"
            stats_text += f"Motion Area: {motion_stats['percentage']:.1f}%\n"
        
        if stats['tracking_stats']:
            tracking_stats = stats['tracking_stats']
            stats_text += f"Tracking: {'Yes' if tracking_stats['is_tracking'] else 'No'}\n"
            stats_text += f"Detection: {tracking_stats['detection_type']}\n"
        
        self.stats_text.setText(stats_text)
    
    def handle_error(self, error_msg):
        """Handle processing errors"""
        self.status_label.setText(f"Status: Error - {error_msg}")
        print(f"Video processing error: {error_msg}")
    
    def toggle_autofocus(self, state):
        """Toggle autofocus module"""
        self.video_processor.autofocus_enabled = state == QtCoreQt.CheckState.Checked
    
    def toggle_tracking(self, state):
        """Toggle tracking module"""
        self.video_processor.tracking_enabled = state == QtCoreQt.CheckState.Checked
    
    def toggle_motion(self, state):
        """Toggle motion detection module"""
        self.video_processor.motion_enabled = state == QtCoreQt.CheckState.Checked
    
    def update_autofocus_params(self):
        """Update autofocus parameters"""
        padding = self.padding_slider.value()
        alpha = self.alpha_slider.value() / 100.0
        min_crop = self.min_crop_slider.value() / 100.0
        
        self.padding_label.setText(str(padding))
        self.alpha_label.setText(f"{alpha:.2f}")
        self.min_crop_label.setText(f"{min_crop:.2f}")
        
        # Update module parameters
        self.video_processor.autofocus.padding = padding
        self.video_processor.autofocus.alpha = alpha
        self.video_processor.autofocus.min_crop_ratio = min_crop
    
    def update_tracking_params(self):
        """Update tracking parameters"""
        smoothing = self.track_smooth_slider.value() / 100.0
        zoom = self.zoom_slider.value() / 100.0
        
        self.track_smooth_label.setText(f"{smoothing:.2f}")
        self.zoom_label.setText(f"{zoom:.2f}")
        
        # Update module parameters
        self.video_processor.tracker.set_smoothing_factor(smoothing)
        self.video_processor.tracker.set_zoom_factor(zoom)
    
    def update_tracking_mode(self, mode):
        """Update tracking mode"""
        self.video_processor.tracker.tracking_mode = mode
        # Reinitialize tracker with new mode
        self.video_processor.tracker = ObjectTracker(tracking_mode=mode)
    
    def update_motion_params(self):
        """Update motion detection parameters"""
        sensitivity = self.sensitivity_slider.value()
        min_area = self.min_area_slider.value()
        
        self.sensitivity_label.setText(str(sensitivity))
        self.min_area_label.setText(str(min_area))
        
        # Update module parameters
        self.video_processor.motion_detector.set_sensitivity(sensitivity)
        self.video_processor.motion_detector.set_min_area(min_area)
    
    def check_ai_fallback(self):
        """Cek apakah AI fallback ke OpenCV dan update status bar"""
        from modules.autofocus import MEDIAPIPE_AVAILABLE as AF_AI
        from modules.tracking import MEDIAPIPE_AVAILABLE as TR_AI
        if not AF_AI or not TR_AI:
            self.status_label.setText(self.status_label.text() + " | AI fallback to OpenCV")
    
    def closeEvent(self, event):
        """Handle application close event with proper cleanup"""
        self.video_processor.stop_capture()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("CameraAI")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = CameraAIMainWindow()
    window.show()
    
    # Start application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()