import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QCheckBox, QSlider, QPushButton,
                           QGroupBox, QGridLayout, QComboBox, QSpinBox, QFrame,
                           QTabWidget, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import time

# Import modules
from camera_handler.webcam import FrameGrabber
from modules.autofocus import AutoFocus
from modules.tracking import ObjectTracker
from modules.motion_detector import MotionDetector


class VideoProcessor(QThread):
    """Thread untuk memproses video secara real-time"""
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    stats_updated = pyqtSignal(dict)
    
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
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
    def start_capture(self):
        """Start video capture"""
        try:
            self.frame_grabber = FrameGrabber(src=self.camera_index).start()
            self.running = True
            self.start()
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_capture(self):
        """Stop video capture"""
        self.running = False
        if self.frame_grabber:
            self.frame_grabber.stop()
        self.quit()
        self.wait()
    
    def run(self):
        """Main processing loop"""
        while self.running:
            if not self.frame_grabber:
                continue
                
            try:
                frame = self.frame_grabber.read()
                if frame is None:
                    continue
                
                # Process frame through enabled modules
                processed_frame = self.process_frame(frame)
                
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
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def process_frame(self, frame):
        """Process frame through all enabled modules"""
        processed = frame.copy()
        
        # Apply autofocus
        if self.autofocus_enabled:
            processed = self.autofocus.process(processed)
        
        # Apply tracking
        if self.tracking_enabled:
            processed = self.tracker.process(processed)
        
        # Apply motion detection
        if self.motion_enabled:
            processed = self.motion_detector.process(processed)
        
        return processed
    
    def emit_stats(self):
        """Emit current statistics"""
        stats = {
            'fps': self.current_fps,
            'autofocus_active': self.autofocus_enabled,
            'tracking_active': self.tracking_enabled and self.tracker.is_tracking,
            'motion_stats': self.motion_detector.get_motion_stats() if self.motion_enabled else None
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
        
        # Setup UI
        self.setup_ui()
        
        # Start video capture
        self.start_camera()
    
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
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera Loading...")
        video_layout.addWidget(self.video_label)
        
        # Status bar
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_layout.addWidget(self.fps_label)
        
        self.status_label = QLabel("Status: Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Camera controls
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in range(5)])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        status_layout.addWidget(QLabel("Camera:"))
        status_layout.addWidget(self.camera_combo)
        
        video_layout.addWidget(status_widget)
        
        main_layout.addWidget(video_widget, 2)
    
    def setup_control_panel(self, main_layout):
        """Setup control panel dengan tabs"""
        control_widget = QWidget()
        control_widget.setMaximumWidth(400)
        control_layout = QVBoxLayout(control_widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # AutoFocus tab
        self.setup_autofocus_tab()
        
        # Tracking tab
        self.setup_tracking_tab()
        
        # Motion Detection tab
        self.setup_motion_tab()
        
        # Stats tab
        self.setup_stats_tab()
        
        control_layout.addWidget(self.tab_widget)
        
        # Global controls
        self.setup_global_controls(control_layout)
        
        main_layout.addWidget(control_widget, 1)
    
    def setup_autofocus_tab(self):
        """Setup AutoFocus control tab"""
        autofocus_widget = QWidget()
        layout = QVBoxLayout(autofocus_widget)
        
        # Enable checkbox
        self.autofocus_checkbox = QCheckBox("Enable AutoFocus")
        self.autofocus_checkbox.stateChanged.connect(self.toggle_autofocus)
        layout.addWidget(self.autofocus_checkbox)
        
        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)
        
        # Padding slider
        params_layout.addWidget(QLabel("Padding:"), 0, 0)
        self.padding_slider = QSlider(Qt.Horizontal)
        self.padding_slider.setRange(0, 200)
        self.padding_slider.setValue(50)
        self.padding_slider.valueChanged.connect(self.update_autofocus_params)
        params_layout.addWidget(self.padding_slider, 0, 1)
        self.padding_label = QLabel("50")
        params_layout.addWidget(self.padding_label, 0, 2)
        
        # Alpha (smoothing) slider
        params_layout.addWidget(QLabel("Smoothing:"), 1, 0)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(40)
        self.alpha_slider.valueChanged.connect(self.update_autofocus_params)
        params_layout.addWidget(self.alpha_slider, 1, 1)
        self.alpha_label = QLabel("0.40")
        params_layout.addWidget(self.alpha_label, 1, 2)
        
        # Min crop ratio slider
        params_layout.addWidget(QLabel("Min Crop:"), 2, 0)
        self.min_crop_slider = QSlider(Qt.Horizontal)
        self.min_crop_slider.setRange(20, 100)
        self.min_crop_slider.setValue(60)
        self.min_crop_slider.valueChanged.connect(self.update_autofocus_params)
        params_layout.addWidget(self.min_crop_slider, 2, 1)
        self.min_crop_label = QLabel("0.60")
        params_layout.addWidget(self.min_crop_label, 2, 2)
        
        layout.addWidget(params_group)
        layout.addStretch()
        
        self.tab_widget.addTab(autofocus_widget, "AutoFocus")
    
    def setup_tracking_tab(self):
        """Setup Tracking control tab"""
        tracking_widget = QWidget()
        layout = QVBoxLayout(tracking_widget)
        
        # Enable checkbox
        self.tracking_checkbox = QCheckBox("Enable Tracking")
        self.tracking_checkbox.stateChanged.connect(self.toggle_tracking)
        layout.addWidget(self.tracking_checkbox)
        
        # Tracking mode
        mode_group = QGroupBox("Tracking Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems(["face", "hand"])
        self.tracking_mode_combo.currentTextChanged.connect(self.update_tracking_mode)
        mode_layout.addWidget(self.tracking_mode_combo)
        
        layout.addWidget(mode_group)
        
        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)
        
        # Smoothing slider
        params_layout.addWidget(QLabel("Smoothing:"), 0, 0)
        self.track_smooth_slider = QSlider(Qt.Horizontal)
        self.track_smooth_slider.setRange(0, 100)
        self.track_smooth_slider.setValue(30)
        self.track_smooth_slider.valueChanged.connect(self.update_tracking_params)
        params_layout.addWidget(self.track_smooth_slider, 0, 1)
        self.track_smooth_label = QLabel("0.30")
        params_layout.addWidget(self.track_smooth_label, 0, 2)
        
        # Zoom slider
        params_layout.addWidget(QLabel("Zoom:"), 1, 0)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_tracking_params)
        params_layout.addWidget(self.zoom_slider, 1, 1)
        self.zoom_label = QLabel("1.00")
        params_layout.addWidget