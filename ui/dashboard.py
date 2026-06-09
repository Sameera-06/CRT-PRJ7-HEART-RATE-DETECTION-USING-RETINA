# ui/dashboard.py
"""
UI Dashboard rendering module.
Draws window layout, custom black plots with axis lines, and START/STOP buttons.
"""

import cv2
import numpy as np

class DashboardRenderer:
    """
    Renders the custom 1000x520 dual-column dashboard.
    Implements black background plots (Signal, Accuracy) and START/STOP button styling.
    """
    def __init__(self, window_width=1000, window_height=520, left_width=600, left_height=450, buffer_size=180):
        self.window_width = window_width
        self.window_height = window_height
        self.left_width = left_width
        self.left_height = left_height
        self.buffer_size = buffer_size
        
        # Colors (BGR)
        self.COLOR_WINDOW_BG = (240, 240, 240)  # Light gray (matching photo)
        self.COLOR_PLOT_BG = (0, 0, 0)          # Black
        self.COLOR_PLOT_LINE = (0, 255, 0)      # Neon green
        self.COLOR_PLOT_AXIS = (255, 255, 255)  # White
        self.COLOR_BLUE_BOX = (255, 0, 0)       # Blue
        self.COLOR_TEXT_BLACK = (0, 0, 0)
        self.COLOR_TEXT_WHITE = (255, 255, 255)
        self.COLOR_WARNING = (0, 0, 255)        # Red

    def draw_custom_plot(self, title_text, signal_buffer, is_paused):
        """Draw a black plot canvas with white coordinates, ticks, and green signal line."""
        plot_canvas = np.zeros((140, 300, 3), dtype=np.uint8)
        plot_canvas[:] = self.COLOR_PLOT_BG
        
        # 1. Centered Title
        text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = (300 - text_size[0]) // 2
        cv2.putText(plot_canvas, title_text, (tx, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        
        # 2. White Axes lines
        cv2.line(plot_canvas, (40, 25), (40, 115), self.COLOR_PLOT_AXIS, 1)
        cv2.line(plot_canvas, (40, 115), (280, 115), self.COLOR_PLOT_AXIS, 1)
        
        # 3. Axis Tick lines
        cv2.line(plot_canvas, (35, 25), (40, 25), self.COLOR_PLOT_AXIS, 1)
        cv2.line(plot_canvas, (35, 115), (40, 115), self.COLOR_PLOT_AXIS, 1)
        
        ticks_x = [40, 146, 200, 280]
        for tx in ticks_x:
            cv2.line(plot_canvas, (tx, 115), (tx, 120), self.COLOR_PLOT_AXIS, 1)
            
        # 4. Text Labels
        cv2.putText(plot_canvas, "1", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(plot_canvas, "0", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        
        cv2.putText(plot_canvas, "0", (37, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(plot_canvas, "80", (138, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(plot_canvas, "120", (190, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(plot_canvas, "180", (270, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        
        # 5. Neon green plot line
        if len(signal_buffer) > 1:
            sig = np.array(signal_buffer)
            min_v = np.min(sig)
            max_v = np.max(sig)
            range_v = max_v - min_v
            if range_v < 1e-5:
                range_v = 1e-5
                
            points = []
            for i, val in enumerate(signal_buffer):
                x = int(40 + (i / (self.buffer_size - 1)) * 240)
                norm_val = (val - min_v) / range_v
                y = int(115 - norm_val * 90)
                points.append((x, y))
                
            line_color = self.COLOR_PLOT_LINE if not is_paused else (100, 100, 100)
            for i in range(len(points) - 1):
                cv2.line(plot_canvas, points[i], points[i+1], line_color, 2, cv2.LINE_AA)
                
        if is_paused:
            cv2.putText(plot_canvas, "PAUSED", (115, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WARNING, 2, cv2.LINE_AA)
            
        return plot_canvas

    def render(self, frame_clean, cropped_eyes, freq_val, bpm_val, is_running, is_paused, pause_reason, filtered_signal, mouse_x, mouse_y):
        """Assemble all sections into the final 1000x520 frame canvas."""
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        canvas[:] = self.COLOR_WINDOW_BG
        
        # 1. Webcam pane
        pane_left = cv2.resize(frame_clean, (self.left_width, self.left_height))
        canvas[20:470, 20:620] = pane_left
        
        # 2. Eye crop sub-view
        border_color = self.COLOR_BLUE_BOX if is_running else (150, 150, 150)
        if cropped_eyes is not None and cropped_eyes.size > 0:
            cropped_resized = cv2.resize(cropped_eyes, (120, 80))
        else:
            cropped_resized = np.zeros((80, 120, 3), dtype=np.uint8)
            cropped_resized[:] = self.COLOR_PLOT_BG
            
        canvas[20:100, 660:780] = cropped_resized
        cv2.rectangle(canvas, (660, 20), (780, 100), border_color, 2)
        
        # Numeric labels next to eye crop
        if not is_paused:
            freq_text = f"Freq: {freq_val:.2f}"
            hr_text = f"Heart rate: {bpm_val:.0f} bpm"
        else:
            freq_text = "Freq: --"
            hr_text = "Heart rate: -- bpm"
            
        cv2.putText(canvas, freq_text, (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT_BLACK, 2, cv2.LINE_AA)
        cv2.putText(canvas, hr_text, (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT_BLACK, 2, cv2.LINE_AA)
        
        # Display text warning reasons
        if is_paused and pause_reason:
            cv2.putText(canvas, pause_reason, (800, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WARNING, 1, cv2.LINE_AA)
            
        # 3. Middle Plot (Signal)
        plot_signal = self.draw_custom_plot("Signal", filtered_signal, is_paused)
        canvas[130:270, 660:960] = plot_signal
        
        # 4. Bottom Plot (Accuracy)
        plot_accuracy = self.draw_custom_plot("Accuracy", filtered_signal, is_paused)
        canvas[300:440, 660:960] = plot_accuracy
        
        # 5. Bottom bar Press ESC label
        cv2.putText(canvas, "Press ESC to Exit", (20, 502), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_BLACK, 1, cv2.LINE_AA)
        
        # Hover colors calculations
        start_hover = 700 <= mouse_x <= 800 and 480 <= mouse_y <= 510
        start_btn_color = (66, 224, 133) if start_hover else (46, 204, 113)
        
        stop_hover = 830 <= mouse_x <= 930 and 480 <= mouse_y <= 510
        stop_btn_color = (80, 96, 251) if stop_hover else (60, 76, 231)
        
        # Draw START
        cv2.rectangle(canvas, (700, 480), (800, 510), start_btn_color, -1)
        cv2.putText(canvas, "START", (722, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
        
        # Draw STOP
        cv2.rectangle(canvas, (830, 480), (930, 510), stop_btn_color, -1)
        cv2.putText(canvas, "STOP", (858, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
        
        return canvas
