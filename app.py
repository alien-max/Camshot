import sys, cv2, os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from datetime import datetime
import mediapipe as mp
from mediapipe import Image
from mediapipe import ImageFormat
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FACE_DETECTOR_PATH = f'{BASE_DIR}/face_detector.tflite'

class Camshot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camshot")
        self.showMaximized()
        self.selected_filter = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.create_main_section(main_layout)
        self.setup_camera()
        self.setup_face_detection()

    def setup_face_detection(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=FACE_DETECTOR_PATH),
            running_mode=VisionRunningMode.IMAGE
        )
        self.detector = FaceDetector.create_from_options(options)

    def create_main_section(self, parent_layout):
        main_section = QWidget()
        main_section.setStyleSheet("background-color: #212121;")

        main_layout = QVBoxLayout(main_section)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.camera_label)

        parent_layout.addWidget(main_section, stretch=1)
        
    def setup_camera(self):
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera_label.setText("Error: Camera not found")
            self.camera_label.setStyleSheet("""
                color: white; 
                font-size: 20px;
                background-color: black;
            """)
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret: return

        self.current_frame = frame.copy()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                bw = int(bbox.width)
                bh = int(bbox.height)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        label_width = self.camera_label.width()
        label_height = self.camera_label.height()

        h, w, ch = frame.shape
        aspect_ratio = w / h
        label_aspect = label_width / label_height if label_height > 0 else 1

        if aspect_ratio > label_aspect:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height))
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))

def main():
    app = QApplication(sys.argv)
    window = Camshot()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()