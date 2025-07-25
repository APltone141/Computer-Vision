import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import time
import threading

try:
    import mediapipe as mp
    from mediapipe.python.solutions.hands import Hands
    from mediapipe.python.solutions.drawing_utils import draw_landmarks
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MediaPipe not available: {e}")
    print("Gesture control will be disabled")
    MEDIAPIPE_AVAILABLE = False

class GestureController:
    """
    Hand gesture recognition dan control system
    Mendukung gesture untuk kontrol aplikasi
    """
    
    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.5):
        # Input validation
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0.0 and 1.0")
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize MediaPipe hands
        if MEDIAPIPE_AVAILABLE:
            try:
                self.hands = Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.detection_type = "mediapipe"
            except Exception as e:
                print(f"MediaPipe hands initialization failed: {e}")
                self.hands = None
                self.detection_type = "none"
        else:
            self.hands = None
            self.detection_type = "none"
        
        # Gesture state
        with self._lock:
            self.current_gesture = "none"
            self.gesture_confidence = 0.0
            self.hand_landmarks = None
            self.gesture_history = []
            self.last_gesture_time = time.time()
            
        # Gesture definitions
        self.gesture_definitions = {
            "fist": self._is_fist,
            "open_palm": self._is_open_palm,
            "pointing": self._is_pointing,
            "thumbs_up": self._is_thumbs_up,
            "thumbs_down": self._is_thumbs_down,
            "peace": self._is_peace,
            "ok": self._is_ok,
            "rock_on": self._is_rock_on
        }
        
        # Command mapping
        self.command_mapping = {
            "fist": "toggle_autofocus",
            "open_palm": "toggle_tracking",
            "pointing": "toggle_motion",
            "thumbs_up": "increase_zoom",
            "thumbs_down": "decrease_zoom",
            "peace": "take_screenshot",
            "ok": "start_recording",
            "rock_on": "stop_recording"
        }
        
        # Gesture cooldown (seconds)
        self.gesture_cooldown = 1.0
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process frame untuk gesture recognition
        
        Returns:
            Tuple[frame, gesture_data]
        """
        if frame is None or frame.size == 0:
            return frame, {}
        
        if self.detection_type != "mediapipe" or self.hands is None:
            return frame, {}
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            gesture_data = {}
            
            if results.multi_hand_landmarks:
                # Process each hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Recognize gesture
                    gesture, confidence = self._recognize_gesture(hand_landmarks)
                    
                    if gesture != "none" and confidence > 0.7:
                        # Update gesture state
                        with self._lock:
                            self.current_gesture = gesture
                            self.gesture_confidence = confidence
                            self.hand_landmarks = hand_landmarks
                            self.gesture_history.append({
                                'gesture': gesture,
                                'confidence': confidence,
                                'time': time.time()
                            })
                            
                            # Keep only last 10 gestures
                            if len(self.gesture_history) > 10:
                                self.gesture_history.pop(0)
                        
                        # Get command
                        command = self.command_mapping.get(gesture, "none")
                        
                        gesture_data = {
                            'gesture': gesture,
                            'confidence': confidence,
                            'command': command,
                            'hand_landmarks': hand_landmarks
                        }
                        
                        # Draw gesture info
                        frame = self._draw_gesture_info(frame, gesture, confidence, command)
                        break
            
            return frame, gesture_data
            
        except Exception as e:
            print(f"Gesture processing error: {e}")
            return frame, {}
    
    def _recognize_gesture(self, landmarks) -> Tuple[str, float]:
        """Recognize gesture from hand landmarks"""
        try:
            # Extract key points
            wrist = landmarks.landmark[0]
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            middle_tip = landmarks.landmark[12]
            ring_tip = landmarks.landmark[16]
            pinky_tip = landmarks.landmark[20]
            
            # Calculate distances and angles
            gestures = {}
            
            # Check each gesture
            for gesture_name, gesture_func in self.gesture_definitions.items():
                confidence = gesture_func(landmarks)
                gestures[gesture_name] = confidence
            
            # Return best gesture
            if gestures:
                best_gesture = max(gestures.items(), key=lambda x: x[1])
                return best_gesture[0], best_gesture[1]
            
            return "none", 0.0
            
        except Exception as e:
            print(f"Gesture recognition error: {e}")
            return "none", 0.0
    
    def _is_fist(self, landmarks) -> float:
        """Detect fist gesture"""
        try:
            # Check if all fingers are closed
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            finger_mids = [3, 7, 11, 15, 19]   # Mid joints
            
            closed_fingers = 0
            for tip, mid in zip(finger_tips, finger_mids):
                tip_y = landmarks.landmark[tip].y
                mid_y = landmarks.landmark[mid].y
                if tip_y > mid_y:  # Finger is closed
                    closed_fingers += 1
            
            confidence = closed_fingers / 5.0
            return confidence if confidence > 0.8 else 0.0
            
        except:
            return 0.0
    
    def _is_open_palm(self, landmarks) -> float:
        """Detect open palm gesture"""
        try:
            # Check if all fingers are open
            finger_tips = [4, 8, 12, 16, 20]
            finger_mids = [3, 7, 11, 15, 19]
            
            open_fingers = 0
            for tip, mid in zip(finger_tips, finger_mids):
                tip_y = landmarks.landmark[tip].y
                mid_y = landmarks.landmark[mid].y
                if tip_y < mid_y:  # Finger is open
                    open_fingers += 1
            
            confidence = open_fingers / 5.0
            return confidence if confidence > 0.8 else 0.0
            
        except:
            return 0.0
    
    def _is_pointing(self, landmarks) -> float:
        """Detect pointing gesture (index finger only)"""
        try:
            # Check if only index finger is extended
            finger_tips = [4, 8, 12, 16, 20]
            finger_mids = [3, 7, 11, 15, 19]
            
            extended_fingers = []
            for tip, mid in zip(finger_tips, finger_mids):
                tip_y = landmarks.landmark[tip].y
                mid_y = landmarks.landmark[mid].y
                if tip_y < mid_y:  # Finger is extended
                    extended_fingers.append(True)
                else:
                    extended_fingers.append(False)
            
            # Only index finger should be extended
            if extended_fingers[1] and not any(extended_fingers[:1] + extended_fingers[2:]):
                return 0.9
            return 0.0
            
        except:
            return 0.0
    
    def _is_thumbs_up(self, landmarks) -> float:
        """Detect thumbs up gesture"""
        try:
            # Check if thumb is up and other fingers are closed
            thumb_tip = landmarks.landmark[4]
            thumb_ip = landmarks.landmark[3]
            
            # Thumb should be pointing up
            if thumb_tip.y < thumb_ip.y:
                # Check if other fingers are closed
                other_fingers = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
                other_mids = [7, 11, 15, 19]
                
                closed_count = 0
                for tip, mid in zip(other_fingers, other_mids):
                    tip_y = landmarks.landmark[tip].y
                    mid_y = landmarks.landmark[mid].y
                    if tip_y > mid_y:  # Finger is closed
                        closed_count += 1
                
                if closed_count >= 3:  # At least 3 other fingers closed
                    return 0.8
            return 0.0
            
        except:
            return 0.0
    
    def _is_thumbs_down(self, landmarks) -> float:
        """Detect thumbs down gesture"""
        try:
            # Check if thumb is down and other fingers are closed
            thumb_tip = landmarks.landmark[4]
            thumb_ip = landmarks.landmark[3]
            
            # Thumb should be pointing down
            if thumb_tip.y > thumb_ip.y:
                # Check if other fingers are closed
                other_fingers = [8, 12, 16, 20]
                other_mids = [7, 11, 15, 19]
                
                closed_count = 0
                for tip, mid in zip(other_fingers, other_mids):
                    tip_y = landmarks.landmark[tip].y
                    mid_y = landmarks.landmark[mid].y
                    if tip_y > mid_y:  # Finger is closed
                        closed_count += 1
                
                if closed_count >= 3:  # At least 3 other fingers closed
                    return 0.8
            return 0.0
            
        except:
            return 0.0
    
    def _is_peace(self, landmarks) -> float:
        """Detect peace sign (index and middle finger)"""
        try:
            # Check if index and middle fingers are extended
            finger_tips = [4, 8, 12, 16, 20]
            finger_mids = [3, 7, 11, 15, 19]
            
            extended_fingers = []
            for tip, mid in zip(finger_tips, finger_mids):
                tip_y = landmarks.landmark[tip].y
                mid_y = landmarks.landmark[mid].y
                if tip_y < mid_y:  # Finger is extended
                    extended_fingers.append(True)
                else:
                    extended_fingers.append(False)
            
            # Only index and middle fingers should be extended
            if extended_fingers[1] and extended_fingers[2] and not any(extended_fingers[:1] + extended_fingers[3:]):
                return 0.9
            return 0.0
            
        except:
            return 0.0
    
    def _is_ok(self, landmarks) -> float:
        """Detect OK sign (thumb and index finger circle)"""
        try:
            # Check distance between thumb and index finger
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
            # Distance should be small for OK sign
            if distance < 0.1:
                return 0.8
            return 0.0
            
        except:
            return 0.0
    
    def _is_rock_on(self, landmarks) -> float:
        """Detect rock on gesture (index and pinky extended)"""
        try:
            # Check if index and pinky fingers are extended
            finger_tips = [4, 8, 12, 16, 20]
            finger_mids = [3, 7, 11, 15, 19]
            
            extended_fingers = []
            for tip, mid in zip(finger_tips, finger_mids):
                tip_y = landmarks.landmark[tip].y
                mid_y = landmarks.landmark[mid].y
                if tip_y < mid_y:  # Finger is extended
                    extended_fingers.append(True)
                else:
                    extended_fingers.append(False)
            
            # Only index and pinky fingers should be extended
            if extended_fingers[1] and extended_fingers[4] and not any(extended_fingers[:1] + extended_fingers[2:4]):
                return 0.9
            return 0.0
            
        except:
            return 0.0
    
    def _draw_gesture_info(self, frame: np.ndarray, gesture: str, confidence: float, command: str) -> np.ndarray:
        """Draw gesture information on frame"""
        h, w = frame.shape[:2]
        
        # Draw gesture text
        text = f"Gesture: {gesture.upper()}"
        cv2.putText(frame, text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw confidence
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw command
        cmd_text = f"Command: {command}"
        cv2.putText(frame, cmd_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw gesture indicator
        if gesture != "none":
            cv2.circle(frame, (w - 50, 50), 30, (0, 255, 0), -1)
            cv2.putText(frame, gesture[0].upper(), (w - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return frame
    
    def get_current_gesture(self) -> Dict[str, Any]:
        """Get current gesture state"""
        with self._lock:
            return {
                'gesture': self.current_gesture,
                'confidence': self.gesture_confidence,
                'command': self.command_mapping.get(self.current_gesture, "none"),
                'history': self.gesture_history.copy()
            }
    
    def get_command(self) -> Optional[str]:
        """Get current command if gesture is recent"""
        with self._lock:
            current_time = time.time()
            if (self.current_gesture != "none" and 
                current_time - self.last_gesture_time > self.gesture_cooldown):
                command = self.command_mapping.get(self.current_gesture, "none")
                self.last_gesture_time = current_time
                return command
            return None
    
    def set_gesture_cooldown(self, cooldown: float):
        """Set gesture cooldown time"""
        if cooldown < 0:
            raise ValueError("Cooldown must be non-negative")
        with self._lock:
            self.gesture_cooldown = cooldown
    
    def add_custom_gesture(self, name: str, gesture_func, command: str):
        """Add custom gesture recognition"""
        if not callable(gesture_func):
            raise ValueError("gesture_func must be callable")
        
        with self._lock:
            self.gesture_definitions[name] = gesture_func
            self.command_mapping[name] = command
    
    def reset_gesture_state(self):
        """Reset gesture state"""
        with self._lock:
            self.current_gesture = "none"
            self.gesture_confidence = 0.0
            self.hand_landmarks = None
            self.gesture_history.clear()
    
    def is_available(self) -> bool:
        """Check if gesture control is available"""
        return self.detection_type == "mediapipe" and self.hands is not None
