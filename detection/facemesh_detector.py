# detection/facemesh_detector.py
"""
MediaPipe FaceMesh detector wrapper.
Handles facial detection and landmark extraction.
"""

import numpy as np
import mediapipe as mp

class FaceMeshDetector:
    """
    Wraps the MediaPipe Face Mesh model.
    Processes RGB frames, extracts landmarks, and computes the Eye Aspect Ratio (EAR).
    """
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, rgb_frame):
        """Process an RGB frame and return MediaPipe face landmarks."""
        return self.face_mesh.process(rgb_frame)

    def get_pixel_landmarks(self, face_landmarks, width, height):
        """Convert normalized face landmarks to pixel coordinates (x, y)."""
        coords = []
        for lm in face_landmarks.landmark:
            coords.append((int(lm.x * width), int(lm.y * height)))
        return coords

    def calculate_ear(self, landmarks_coords):
        """
        Calculate the Eye Aspect Ratio (EAR) for both eyes.
        EAR = (dist(V1_top, V1_bottom) + dist(V2_top, V2_bottom)) / (2 * dist(H_left, H_right))
        """
        try:
            # Helper for distance
            def dist(p1, p2):
                return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            # Right Eye (MediaPipe indices)
            p33 = landmarks_coords[33]      # Right outer corner
            p133 = landmarks_coords[133]    # Right inner corner
            p159 = landmarks_coords[159]    # Top eyelid 1
            p145 = landmarks_coords[145]    # Bottom eyelid 1
            p158 = landmarks_coords[158]    # Top eyelid 2
            p153 = landmarks_coords[153]    # Bottom eyelid 2
            
            dist_r_h = dist(p33, p133)
            dist_r_v1 = dist(p159, p145)
            dist_r_v2 = dist(p158, p153)
            ear_r = (dist_r_v1 + dist_r_v2) / (2.0 * dist_r_h + 1e-6)
            
            # Left Eye (MediaPipe indices)
            p362 = landmarks_coords[362]    # Left inner corner
            p263 = landmarks_coords[263]    # Left outer corner
            p386 = landmarks_coords[386]    # Top eyelid 1
            p374 = landmarks_coords[374]    # Bottom eyelid 1
            p385 = landmarks_coords[385]    # Top eyelid 2
            p380 = landmarks_coords[380]    # Bottom eyelid 2
            
            dist_l_h = dist(p362, p263)
            dist_l_v1 = dist(p386, p374)
            dist_l_v2 = dist(p385, p380)
            ear_l = (dist_l_v1 + dist_l_v2) / (2.0 * dist_l_h + 1e-6)
            
            return (ear_r + ear_l) / 2.0
        except Exception:
            return 0.3  # Nominal default

    def close(self):
        """Release FaceMesh resources."""
        self.face_mesh.close()
