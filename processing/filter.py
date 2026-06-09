# processing/filter.py
"""
Signal filtering and cleaning module.
Performs detrending and Butterworth bandpass filtering.
"""

import numpy as np
from scipy.signal import butter, filtfilt

class SignalFilter:
    """
    Cleans physiological signals by detrending (removing baseline drift)
    and bandpass filtering (isolating targeted frequencies).
    """
    def __init__(self, min_bpm=50.0, max_bpm=110.0):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def detrend(self, raw_signal, fps):
        """Remove low-frequency light drift by subtracting the rolling local mean."""
        window_size = int(fps)
        if window_size % 2 == 0:
            window_size += 1
            
        rolling_mean = np.convolve(raw_signal, np.ones(window_size)/window_size, mode='same')
        detrended = raw_signal - rolling_mean
        
        # Zero out boundary edges to suppress convolution artifacts
        half_w = window_size // 2
        detrended[:half_w] = 0
        detrended[-half_w:] = 0
        
        return detrended

    def bandpass_filter(self, signal_data, fps):
        """Apply a 4th-order Butterworth bandpass filter tuned to [MIN_BPM, MAX_BPM]."""
        low_cut = self.min_bpm / 60.0
        high_cut = self.max_bpm / 60.0
        nyq = 0.5 * fps
        
        # Normalize bounds relative to Nyquist frequency
        low = max(0.01, min(low_cut / nyq, 0.99))
        high = max(low + 0.01, min(high_cut / nyq, 0.99))
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal_data)
        except Exception:
            filtered = signal_data
            
        return filtered
