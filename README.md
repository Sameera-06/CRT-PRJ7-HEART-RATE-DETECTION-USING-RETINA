# ❤️ Retina Heart Rate AI

A real-time Heart Rate Detection System that estimates heart rate and pulse frequency using eye-retina video signals captured through a webcam.

The system uses MediaPipe Face Mesh for eye detection, CNN-based feature extraction, signal processing, and frequency analysis to estimate heart rate without any wearable sensors.

---

## 📌 Features

* Real-time webcam monitoring
* Eye Retina Region Detection
* MediaPipe Face Mesh based eye localization
* CNN-based feature extraction
* rPPG (Remote Photoplethysmography) signal extraction
* Bandpass signal filtering
* Heart Rate (BPM) estimation
* Frequency estimation
* Live signal visualization
* Accuracy monitoring graph
* User-friendly dashboard interface

---

## 🖥️ Project Demo

### Dashboard

* Webcam Feed
* Retina Detection
* Heart Rate Display
* Frequency Display
* Signal Graph
* Accuracy Graph

---

## 📂 Project Structure

```text
RETINA_HR_AI
│
├── __pycache__
│   └── main.cpython-310.pyc
│
├── cnn
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── efficientphys.cpython-310.pyc
│   ├── __init__.py
│   └── efficientphys.py
│
├── detection
├── outputs
│
├── image.png
│
├── processing
├── ui
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Technologies Used

### Computer Vision

* OpenCV
* MediaPipe

### Deep Learning

* TensorFlow
* Keras
* CNN (EfficientPhys)

### Signal Processing

* NumPy
* SciPy

### Visualization

* OpenCV Dashboard
* Matplotlib

---

## 🔬 Working Principle

### Step 1: Webcam Capture

The webcam continuously captures video frames in real time.

### Step 2: Face and Eye Detection

MediaPipe Face Mesh detects facial landmarks and locates both eyes.

### Step 3: Retina ROI Extraction

Eye regions are extracted from the detected landmarks.

### Step 4: CNN Feature Extraction

EfficientPhys-based CNN extracts physiological features from the retina region.

### Step 5: Signal Generation

The extracted features are converted into a raw pulse signal.

### Step 6: Signal Filtering

A bandpass filter removes noise and preserves physiological frequencies.

### Step 7: Frequency Analysis

FFT (Fast Fourier Transform) identifies the dominant pulse frequency.

### Step 8: Heart Rate Calculation

Heart Rate (BPM) is calculated as:

```text
Heart Rate (BPM) = Frequency × 60
```

### Step 9: Dashboard Display

The dashboard displays:

* Heart Rate
* Frequency
* Signal Graph
* Accuracy Graph

---

## 📊 Output Parameters

### Heart Rate

Measured in Beats Per Minute (BPM).

Example:

```text
Heart Rate: 58 BPM
```

### Frequency

Measured in Hertz (Hz).

Example:

```text
Freq: 0.97 Hz
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/retina-heart-rate-ai.git
cd retina-heart-rate-ai
```

### Create Virtual Environment

```bash
python -m venv env
```

### Activate Environment

#### Windows

```bash
env\Scripts\activate
```

#### Linux / Mac

```bash
source env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

```bash
python main.py
```

<img width="997" height="543" alt="Screenshot 2026-06-08 093532" src="https://github.com/user-attachments/assets/1a9f1d50-7f2d-46f9-acf6-e4bfd885b096" />


---

## 🎯 Applications

* Contactless Heart Rate Monitoring
* Telemedicine
* Healthcare Research
* Smart Health Systems
* Remote Patient Monitoring
* Wellness Tracking

---

## 🔮 Future Enhancements

* Multi-person Monitoring
* Stress Level Detection
* Blood Pressure Estimation
* SpO₂ Estimation
* Mobile Application Integration
* Cloud-based Health Dashboard

---

## 👨‍💻 Author

SAMEERA.SHAIK.

---

## 📜 License

This project is developed for educational and research purposes.
