# processing/blink_detector.py
"""
Blink and eye state detector.
Tracks Eye Aspect Ratio (EAR) history to detect eye closure.
"""

class BlinkDetector:
    """
    Classifies the state of the eyelids (open or closed) based on the computed EAR
    and a minimum frame persistence duration.
    """
    def __init__(self, ear_threshold=0.20, consec_frames=3):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.closed_counter = 0

    def check_eye_state(self, ear):
        """
        Evaluate current EAR against threshold.
        Returns True if eyes are closed (below threshold for consec_frames), False otherwise.
        """
        if ear < self.ear_threshold:
            self.closed_counter += 1
            if self.closed_counter >= self.consec_frames:
                return True
        else:
            self.closed_counter = 0
            
        return False
