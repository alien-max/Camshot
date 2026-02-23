# Camshot 📸

A simple camera application with real-time face detection built with Python.

## Features

- 🎥 Real-time camera feed
- 👤 Face detection using MediaPipe
- 🖥️ Modern dark-themed UI

## Screenshots

The application displays a live camera feed with face detection boxes drawn around detected faces.

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/camshot.git
cd camshot
```

2. Install required packages:
```bash
pip install opencv-python PySide6 mediapipe
```

3. Make sure you have the `face_detector.tflite` model file in the same directory as the script.

## Usage

Run the application:
```bash
python app.py
```

## How It Works

The application uses:
- **OpenCV** for camera capture and image processing
- **PySide6** for the graphical user interface
- **MediaPipe** for real-time face detection
