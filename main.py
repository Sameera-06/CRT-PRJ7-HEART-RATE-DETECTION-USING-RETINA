# main.py
"""
Main application script for Retina Heart Rate AI.
Integrates facial and hand detection, signal filtering, BPM estimation, and dashboard UI.
"""

import cv2
import numpy as np
import time

# Import modular components
from cnn.efficientphys import EfficientPhysExtractor
from detection.facemesh_detector import FaceMeshDetector
from detection.eye_roi import EyeROIExtractor
from processing.blink_detector import BlinkDetector
from processing.filter import SignalFilter
from processing.bpm import BPMEstimator
from ui.dashboard import DashboardRenderer

# Globals for application states
is_running = True
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    """Callback function to capture mouse coordinate positions and handle clicks in OpenCV window."""
    global is_running, mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check if START clicked: x in [700, 800], y in [480, 510]
        if 700 <= x <= 800 and 480 <= y <= 510:
            is_running = True
            print("START button clicked: Real-time detection active.")
        # Check if STOP clicked: x in [830, 930], y in [480, 510]
        elif 830 <= x <= 930 and 480 <= y <= 510:
            is_running = False
            print("STOP button clicked: Real-time detection paused.")

def main():
    global is_running, mouse_x, mouse_y
    
    # Configuration parameters
    CAMERA_INDEX = 0
    INPUT_WIDTH, INPUT_HEIGHT = 640, 480
    BUFFER_SIZE = 180
    EAR_THRESHOLD = 0.20
    MIN_BPM, MAX_BPM = 50.0, 110.0
    
    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not access webcam index {CAMERA_INDEX}.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)
    
    # Instantiate modular components
    extractor = EfficientPhysExtractor()
    detector = FaceMeshDetector()
    roi_extractor = EyeROIExtractor()
    blink_detector = BlinkDetector(ear_threshold=EAR_THRESHOLD, consec_frames=3)
    sig_filter = SignalFilter(min_bpm=MIN_BPM, max_bpm=MAX_BPM)
    bpm_estimator = BPMEstimator(min_bpm=MIN_BPM, max_bpm=MAX_BPM)
    renderer = DashboardRenderer(buffer_size=BUFFER_SIZE)
    
    # Timing and signal buffers
    fps_calc_start = time.time()
    fps_frames = 0
    fps = 30.0
    
    raw_buffer = []
    filtered_buffer = []
    
    # Create window and set mouse events callback
    window_title = "RETINA HEART RATE AI"
    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, mouse_callback)
    
    print("Launching Modular RETINA HEART RATE AI...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture webcam frame.")
            break
            
        frame_clean = frame.copy()
        
        # Calculate dynamic capture FPS
        fps_frames += 1
        elapsed = time.time() - fps_calc_start
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_frames = 0
            fps_calc_start = time.time()
            
        # Convert color for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh landmarks
        mesh_results = detector.process(rgb_frame)
        face_detected = mesh_results.multi_face_landmarks is not None
        
        # Default placeholder values
        cropped_eyes = None
        freq_val = 0.0
        bpm_val = 0.0
        eyes_closed = False
        hands_near_eyes = False
        pause_reason = ""
        
        if face_detected:
            # 1. Get pixel coordinates
            landmarks_coords = detector.get_pixel_landmarks(mesh_results.multi_face_landmarks[0], INPUT_WIDTH, INPUT_HEIGHT)
            
            # 2. Check Eyelid closure
            current_ear = detector.calculate_ear(landmarks_coords)
            eyes_closed = blink_detector.check_eye_state(current_ear)
            
            # 3. Get eye box and crop
            eye_box = roi_extractor.get_eye_box(landmarks_coords, INPUT_WIDTH, INPUT_HEIGHT)
            cropped_eyes = roi_extractor.crop_eyes(frame_clean, eye_box)
            
            # 4. Check hand occlusion
            hands_near_eyes = roi_extractor.check_hand_occlusion(rgb_frame, eye_box, INPUT_WIDTH, INPUT_HEIGHT)
            
            # Draw blue bounding box on webcam view (or gray if stopped)
            box_color = renderer.COLOR_BLUE_BOX if is_running else (150, 150, 150)
            cv2.rectangle(frame, (eye_box[0], eye_box[1]), (eye_box[2], eye_box[3]), box_color, 2)
            
            # 5. Handle signal updates and pause triggers
            is_paused = not is_running or eyes_closed or hands_near_eyes
            
            if is_paused:
                if not is_running:
                    pause_reason = "PAUSED (Stopped)"
                elif eyes_closed:
                    pause_reason = "PAUSED (Eyes Closed)"
                else:
                    pause_reason = "PAUSED (Hands Occluding)"
                
                # Keep graphs frozen
                bpm_val = 0.0
                freq_val = 0.0
            else:
                # Active detection: extract color signal
                pulse_value = extractor.extract_signal(cropped_eyes)
                raw_buffer.append(pulse_value)
                if len(raw_buffer) > BUFFER_SIZE:
                    raw_buffer.pop(0)
                    
                # Process signal once buffer is filled
                if len(raw_buffer) >= BUFFER_SIZE:
                    detrended = sig_filter.detrend(np.array(raw_buffer), fps)
                    filtered = sig_filter.bandpass_filter(detrended, fps)
                    filtered_buffer = list(filtered)
                    
                    # Estimate BPM and Frequency
                    bpm_val, freqs, mags = bpm_estimator.estimate(filtered, fps)
                    freq_val = bpm_val / 60.0
                else:
                    pause_reason = f"CALIBRATING ({len(raw_buffer)}/{BUFFER_SIZE})"
                    is_paused = True
                    bpm_val = 0.0
                    freq_val = 0.0
        else:
            # Face not detected
            is_paused = True
            pause_reason = "NO FACE DETECTED"
            bpm_val = 0.0
            freq_val = 0.0
            
        # Draw UI
        window_img = renderer.render(
            frame_clean=frame,
            cropped_eyes=cropped_eyes,
            freq_val=freq_val,
            bpm_val=bpm_val,
            is_running=is_running,
            is_paused=is_paused,
            pause_reason=pause_reason,
            filtered_signal=filtered_buffer,
            mouse_x=mouse_x,
            mouse_y=mouse_y
        )
        
        cv2.imshow(window_title, window_img)
        
        # Quit key triggers
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
            
    # Release resources
    cap.release()
    detector.close()
    roi_extractor.close()
    cv2.destroyAllWindows()
    print("Application shut down successfully.")

if __name__ == "__main__":
    main()
