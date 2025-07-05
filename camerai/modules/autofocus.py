import cv2
import mediapipe as mp

class AutoFocus:
    def __init__(self, min_confidence=0.6, padding=50, alpha=0.4, min_crop_ratio=0.6):
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)
        self.padding = padding
        self.alpha = alpha  # smoothing factor
        self.min_crop_ratio = min_crop_ratio
        self.prev_bbox = None  # (x, y, w, h)

    def process(self, frame):
        ih, iw, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if results.detections:
            best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width *
                                                      d.location_data.relative_bounding_box.height)
            bbox = best.location_data.relative_bounding_box

            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            # smooth: EMA
            if self.prev_bbox:
                x = int(self.alpha * x + (1 - self.alpha) * self.prev_bbox[0])
                y = int(self.alpha * y + (1 - self.alpha) * self.prev_bbox[1])
                w = int(self.alpha * w + (1 - self.alpha) * self.prev_bbox[2])
                h = int(self.alpha * h + (1 - self.alpha) * self.prev_bbox[3])

            self.prev_bbox = (x, y, w, h)

        elif self.prev_bbox:
            # pakai previous bbox jika deteksi hilang
            x, y, w, h = self.prev_bbox
        else:
            # tidak ada deteksi sama sekali
            return frame

        # enforce min crop area
        min_w, min_h = int(iw * self.min_crop_ratio), int(ih * self.min_crop_ratio)
        w = max(w, min_w)
        h = max(h, min_h)

        # padding
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(iw, x + w + self.padding)
        y2 = min(ih, y + h + self.padding)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame

        focused = cv2.resize(cropped, (iw, ih), interpolation=cv2.INTER_CUBIC)
        return focused
