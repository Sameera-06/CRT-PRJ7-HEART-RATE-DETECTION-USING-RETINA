# processing/bpm.py
"""
BPM estimation module.
Performs Fast Fourier Transform (FFT) to extract heart rate frequency.
"""

import numpy as np

class BPMEstimator:
    """
    Estimates heart rate in Beats Per Minute (BPM) by detecting peak frequencies
    in physiological signals using real-valued FFT.
    """
    def __init__(self, min_bpm=50.0, max_bpm=110.0):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.last_bpm = 72.0

    def estimate(self, filtered_signal, fps):
        """
        Estimate the current BPM from the filtered signal buffer.
        Returns: (bpm, fft_freqs, fft_mags)
        """
        n = len(filtered_signal)
        fft_out = np.fft.rfft(filtered_signal)
        fft_freqs = np.fft.rfftfreq(n, d=1.0/fps)
        fft_mags = np.abs(fft_out)
        
        low_cut = self.min_bpm / 60.0
        high_cut = self.max_bpm / 60.0
        
        freq_mask = (fft_freqs >= low_cut) & (fft_freqs <= high_cut)
        
        if np.any(freq_mask):
            valid_freqs = fft_freqs[freq_mask]
            valid_mags = fft_mags[freq_mask]
            
            # Identify frequency bin with maximum power
            peak_idx = np.argmax(valid_mags)
            peak_freq = valid_freqs[peak_idx]
            
            detected_bpm = peak_freq * 60.0
            
            # Constrain heart rate strictly: >50 and <110 BPM
            detected_bpm = max(self.min_bpm + 0.5, min(self.max_bpm - 0.5, detected_bpm))
            
            # Apply low-pass smoothing to stabilize readings
            alpha = 0.20
            self.last_bpm = alpha * detected_bpm + (1.0 - alpha) * self.last_bpm
            
        return self.last_bpm, fft_freqs, fft_mags
pre_compiled_bpm = None
