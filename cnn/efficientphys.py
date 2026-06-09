# cnn/efficientphys.py
"""
EfficientPhys-inspired rPPG signal extractor.
Extracts cardiac signal values from the spatial-temporal color channels of the eye ROI.
"""

import numpy as np

class EfficientPhysExtractor:
    """
    Extracts robust remote photoplethysmography (rPPG) signals from the cropped eyes ROI
    using a spatial-temporal Green channel and chrominance projection.
    """
    def __init__(self):
        pass

    def extract_signal(self, cropped_eyes):
        """
        Extract the raw physiological pulse signal value from the cropped eye crop.
        Green channel has the highest light absorption variations due to oxygenated hemoglobin.
        """
        if cropped_eyes is None or cropped_eyes.size == 0:
            return 0.0
        
        # Spatial average of the Green channel (index 1 in OpenCV BGR)
        mean_green = np.mean(cropped_eyes[:, :, 1])
        
        # Spatial average of the Red channel (index 2 in OpenCV BGR)
        mean_red = np.mean(cropped_eyes[:, :, 2])
        
        # Chrominance combination to reduce common-mode motion artifacts
        pulse_value = mean_green - 0.12 * mean_red
        
        return pulse_value
