# detection/eye_roi.py
"""
Eye Region of Interest (ROI) and hand occlusion detector.
Calculates coordinates, handles crops, and monitors hand overlap.
"""

import cv2
import numpy as np
import mediapipe as mp

class EyeROIExtractor:
    """
    Computes the bounding box encompassing both eyes, performs cropping,
    and runs hand detection to monitor hand-over-eyes occlusion.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Key landmark indices (eyes + iris)
        self.eye_landmarks_indices = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

    def get_eye_box(self, landmarks_coords, width, height):
        """Calculate the bounding box enclosing both eyes in pixel coordinates."""
        try:
            eye_xs = [landmarks_coords[idx][0] for idx in self.eye_landmarks_indices]
            eye_ys = [landmarks_coords[idx][1] for idx in self.eye_landmarks_indices]
            min_x, max_x = min(eye_xs), max(eye_xs)
            min_y, max_y = min(eye_ys), max(eye_ys)
            
            eye_w = max_x - min_x
            eye_h = max_y - min_y
            
            # Apply padding to capture surrounding vascularized tissue
            pad_x = int(eye_w * 0.15)
            pad_y = int(eye_h * 0.35)
            
            x1 = max(0, min_x - pad_x)
            y1 = max(0, min_y - pad_y)
            x2 = min(width, max_x + pad_x)
            y2 = min(height, max_y + pad_y)
            
            return x1, y1, x2, y2
        except Exception:
            # Fallback values if landmark indices are out of range
            return 0, 0, width, height

    def crop_eyes(self, frame, box):
        """Crop the eye region from the image frame."""
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]

    def check_hand_occlusion(self, rgb_frame, eye_box, width, height, detect_pad=12):
        """
        Check if any hand is overlapping with the eyes bounding box.
        Returns True if hand is detected near/over the eyes, otherwise False.
        """
        hands_results = self.hands.process(rgb_frame)
        if not hands_results.multi_hand_landmarks:
            return False
        
        x1, y1, x2, y2 = eye_box
        
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                hx, hy = int(lm.x * width), int(lm.y * height)
                # Check bounds including a margin of sensitivity
                if (x1 - detect_pad) <= hx <= (x2 + detect_pad) and \
                   (y1 - detect_pad) <= hy <= (y2 + detect_pad):
                    return True
        return False

    def close(self):
        """Release MediaPipe Hands resources."""
        self.hands.close()
