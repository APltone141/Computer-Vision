# 📁 CameraAI Assets Directory

Direktori ini berisi sample data dan assets untuk testing dan development CameraAI.

## 📂 Struktur Directory

```
assets/
├── test_images/          # Sample images untuk testing
│   ├── faces/           # Face detection test images
│   ├── hands/           # Hand gesture test images
│   └── scenes/          # General scene test images
├── test_videos/         # Sample videos untuk testing
│   ├── short_clips/     # Short video clips
│   └── long_videos/     # Longer video files
├── models/              # Pre-trained models (optional)
│   ├── face_detection/  # Face detection models
│   └── super_resolution/ # Super resolution models
└── README.md           # This file
```

## 🖼️ Test Images

### Face Detection Images
- `faces/person_1.jpg` - Single person portrait
- `faces/person_2.jpg` - Multiple people
- `faces/group.jpg` - Group photo
- `faces/profile.jpg` - Profile view

### Hand Gesture Images
- `hands/fist.jpg` - Fist gesture
- `hands/open_palm.jpg` - Open palm
- `hands/pointing.jpg` - Pointing gesture
- `hands/thumbs_up.jpg` - Thumbs up
- `hands/peace.jpg` - Peace sign

### Scene Images
- `scenes/indoor.jpg` - Indoor scene
- `scenes/outdoor.jpg` - Outdoor scene
- `scenes/low_light.jpg` - Low light condition
- `scenes/motion.jpg` - Scene with motion

## 🎥 Test Videos

### Short Clips (5-30 seconds)
- `short_clips/face_tracking.mp4` - Face tracking test
- `short_clips/hand_gestures.mp4` - Hand gesture recognition
- `short_clips/motion_detection.mp4` - Motion detection test
- `short_clips/autofocus_test.mp4` - Auto-focus test

### Long Videos (1-5 minutes)
- `long_videos/interview.mp4` - Interview scenario
- `long_videos/presentation.mp4` - Presentation scenario
- `long_videos/meeting.mp4` - Meeting scenario

## 🤖 Models (Optional)

### Face Detection Models
- `models/face_detection/haarcascade_frontalface_default.xml`
- `models/face_detection/haarcascade_eye.xml`

### Super Resolution Models
- `models/super_resolution/RealESRGAN_x4plus.pth`
- `models/super_resolution/RealESRGAN_x4plus_anime_6B.pth`

## 🧪 Testing Instructions

### 1. Image Testing
```python
from utils.image_processor import ImageProcessor
from modules.autofocus import AutoFocus
from modules.tracking import ObjectTracker

# Test with sample images
processor = ImageProcessor()
autofocus = AutoFocus()
tracker = ObjectTracker()

# Load test image
test_image = cv2.imread("assets/test_images/faces/person_1.jpg")

# Test processing
processed = processor.process_image(test_image, filter_name="sharpen")
focused = autofocus.process(test_image)
tracked = tracker.process(test_image)
```

### 2. Video Testing
```python
from camera_handler.webcam import FrameGrabber
from utils.video_recorder import VideoRecorder

# Test video recording
recorder = VideoRecorder()
recorder.start_recording("test_output.mp4")

# Process video frames
cap = cv2.VideoCapture("assets/test_videos/short_clips/face_tracking.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame = autofocus.process(frame)
    recorder.write_frame(processed_frame)

cap.release()
recorder.stop_recording()
```

### 3. Performance Testing
```python
import time
from utils.performance import PerformanceMonitor

# Test performance with sample data
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Process multiple images
start_time = time.time()
for i in range(100):
    test_image = cv2.imread(f"assets/test_images/scenes/scene_{i%5}.jpg")
    processed = autofocus.process(test_image)
    monitor.frame_end()

end_time = time.time()
print(f"Processed 100 images in {end_time - start_time:.2f} seconds")
print(monitor.get_formatted_stats())
```

## 📊 Expected Results

### Performance Benchmarks
- **AutoFocus**: 15-30 FPS on CPU, 30-60 FPS on GPU
- **Tracking**: 20-40 FPS on CPU, 40-80 FPS on GPU
- **Motion Detection**: 25-50 FPS on CPU, 50-100 FPS on GPU
- **Super Resolution**: 5-15 FPS on CPU, 15-30 FPS on GPU

### Quality Metrics
- **Face Detection Accuracy**: >90% on test images
- **Hand Gesture Recognition**: >85% on test images
- **Motion Detection Sensitivity**: Configurable 10-200
- **Super Resolution Quality**: PSNR >30dB

## 🔧 Setup Instructions

### 1. Download Sample Data
```bash
# Create directories
mkdir -p assets/test_images/{faces,hands,scenes}
mkdir -p assets/test_videos/{short_clips,long_videos}

# Download sample images (you can use any free stock photos)
# Example: Download from Unsplash or similar free image sites
```

### 2. Generate Test Videos
```python
# Create test videos from images
import cv2
import numpy as np

def create_test_video(output_path, duration=10, fps=30):
    """Create a simple test video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    for i in range(duration * fps):
        # Create frame with moving object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int(320 + 200 * np.sin(i * 0.1))
        y = int(240 + 100 * np.cos(i * 0.1))
        cv2.circle(frame, (x, y), 50, (255, 255, 255), -1)
        out.write(frame)
    
    out.release()

# Create test videos
create_test_video("assets/test_videos/short_clips/motion_detection.mp4")
```

### 3. Model Setup (Optional)
```bash
# Download pre-trained models
mkdir -p assets/models/{face_detection,super_resolution}

# Download OpenCV models
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml -O assets/models/face_detection/

# Download Real-ESRGAN models (if using super resolution)
# Note: These are large files, download only if needed
```

## 📝 Notes

- **Image Formats**: Use JPG/PNG for images, MP4/AVI for videos
- **File Sizes**: Keep test files reasonable (<100MB each)
- **Licensing**: Ensure all sample data is properly licensed
- **Updates**: Update test data periodically to maintain quality

## 🚀 Quick Start

1. **Clone repository** and navigate to assets directory
2. **Download sample data** or create your own test files
3. **Run tests** using the provided scripts
4. **Verify results** against expected benchmarks
5. **Report issues** if performance doesn't meet expectations

---

*Last updated: 2024*
*CameraAI Assets v1.0* 