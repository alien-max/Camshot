import sys
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QCheckBox, QFrame, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from config import SIDEBAR_WIDTH, SNAPSHOTS_DIR
from core import CameraPipeline, Renderer
from analyzer import FaceAnalyzer, EyeTracker, AttributesDNN, HandAnalyzer

class Sidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet("background:#1a1a1a;")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)

        t = QLabel("Camshot")
        t.setFont(QFont("Segoe UI", 15, QFont.Bold))
        t.setStyleSheet("color:#fff; padding-bottom:2px;")
        root.addWidget(t)

        sub = QLabel("Real-time Face Analysis")
        sub.setStyleSheet("color:#555; font-size:11px; padding-bottom:14px;")
        root.addWidget(sub)

        self._add_divider(root, "LIVE STATS")
        self._s_fps    = self._stat_row(root, "FPS")
        self._s_faces  = self._stat_row(root, "Faces")
        self._s_gender = self._stat_row(root, "Gender")
        self._s_age    = self._stat_row(root, "Age")
        self._s_blinks = self._stat_row(root, "Blinks")
        self._s_hands  = self._stat_row(root, "Hands")
        self._s_fingers = self._stat_row(root, "Fingers")

        self._add_divider(root, "OVERLAYS")
        self.tog_bbox  = self._toggle(root, "Bounding box",     True)
        self.tog_mesh  = self._toggle(root, "Face mesh",        False)
        self.tog_lm    = self._toggle(root, "Landmarks",        False)
        self.tog_blur  = self._toggle(root, "Background blur",  False)
        self.tog_pixel = self._toggle(root, "Pixelate faces",   False)

        self._add_divider(root, "ACTIONS")
        self.btn_snap  = self._button(root, "📷  Snapshot",           "#2d5a27", "#3a7a33")
        self.btn_reset = self._button(root, "↺  Reset blink counter", "#2a2a2a", "#3a3a3a")

        root.addStretch()

    def _add_divider(self, layout, title):
        sp = QLabel(); sp.setFixedHeight(10); layout.addWidget(sp)
        lbl = QLabel(title)
        lbl.setStyleSheet("color:#555; font-size:10px; letter-spacing:1px;")
        layout.addWidget(lbl)
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border:none; border-top:1px solid #2a2a2a; margin-bottom:4px;")
        layout.addWidget(line)

    def _stat_row(self, layout, label) -> QLabel:
        row = QWidget()
        h   = QHBoxLayout(row)
        h.setContentsMargins(0, 1, 0, 1)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#666; font-size:12px;")
        lbl.setFixedWidth(80)
        val = QLabel("—")
        val.setStyleSheet("color:#ddd; font-size:12px; font-weight:bold;")
        h.addWidget(lbl); h.addWidget(val); h.addStretch()
        layout.addWidget(row)
        return val

    def _toggle(self, layout, label, default) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(default)
        cb.setStyleSheet("""
            QCheckBox { color:#bbb; font-size:12px; spacing:8px; padding:2px 0; }
            QCheckBox::indicator { width:15px; height:15px; border-radius:3px;border:1px solid #444; background:#222; }
            QCheckBox::indicator:checked { background:#4CAF50; border-color:#4CAF50; }
        """)
        layout.addWidget(cb)
        return cb

    def _button(self, layout, label, bg, hover) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(f"""
            QPushButton {{ background:{bg}; color:#ccc; border:none;border-radius:5px; padding:7px; font-size:12px; margin-top:5px; }}
            QPushButton:hover {{ background:{hover}; }}
        """)
        layout.addWidget(btn)
        return btn

    def update_stats(self, fps, faces, attrs, blinks, hands):
        self._s_fps.setText(f"{fps:.1f}")
        self._s_faces.setText(str(len(faces)))

        if faces:
            att = attrs[0] if attrs else {}
            self._s_gender.setText(f"{att.get('gender','—')}  {att.get('gender_conf',0):.0f}%")
            self._s_age.setText(f"{att.get('age',0)}  {att.get('age_range','—')}")
            self._s_blinks.setText(str(blinks[0] if blinks else 0))
        else:
            for w in [self._s_gender, self._s_age, self._s_blinks]:
                w.setText("—")

        self._s_hands.setText(str(len(hands)))
        total = hands[0]["total"] if hands else 0
        self._s_fingers.setText(str(total))

    def get_flags(self) -> dict:
        return {
            "show_bbox":      self.tog_bbox.isChecked(),
            "show_mesh":      self.tog_mesh.isChecked(),
            "show_landmarks": self.tog_lm.isChecked(),
            "blur_bg":        self.tog_blur.isChecked(),
            "pixelate":       self.tog_pixel.isChecked(),
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camshot")
        self.showMaximized()

        self._pipeline = CameraPipeline()
        self._faces_an = FaceAnalyzer()
        self._hand_an  = HandAnalyzer()
        self._attrs    = AttributesDNN()
        self._renderer = Renderer()
        self._trackers: list[EyeTracker] = []

        self._faces     = []
        self._hands     = []
        self._attr_data = []
        self._fps       = 0.0
        self._last_t    = time.time()

        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._cam = QLabel()
        self._cam.setAlignment(Qt.AlignCenter)
        self._cam.setStyleSheet("background:#000;")
        lay.addWidget(self._cam, stretch=1)

        self._sidebar = Sidebar()
        self._sidebar.btn_snap.clicked.connect(self._snapshot)
        self._sidebar.btn_reset.clicked.connect(self._reset_blinks)
        lay.addWidget(self._sidebar)

        if not self._pipeline.start():
            QMessageBox.critical(self, "Error", "Cannot open camera.")
            sys.exit(1)
        self._attrs.start()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _tick(self):
        frame = self._pipeline.get_frame()
        if frame is None:
            return

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = rgb.shape[:2]

        self._faces = self._faces_an.detect(rgb)
        while len(self._trackers) < len(self._faces):
            self._trackers.append(EyeTracker())
        for i, face in enumerate(self._faces):
            self._trackers[i].update(face["landmark_norm"], w, h)
        blink_counts = [t.total_blinks for t in self._trackers]

        self._hands = self._hand_an.detect(rgb)
        self._attrs.submit(rgb, [f["bbox"] for f in self._faces])
        self._attr_data = self._attrs.get()

        rendered = self._renderer.draw(
            rgb, self._faces, self._attr_data,
            blink_counts, self._hands,
            self._sidebar.get_flags(),
        )

        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now

        self._show(rendered)
        self._sidebar.update_stats(self._fps, self._faces, self._attr_data, blink_counts, self._hands)

    def _show(self, rgb: np.ndarray):
        lw, lh = self._cam.width(), self._cam.height()
        if lw <= 0 or lh <= 0:
            return
        fh, fw = rgb.shape[:2]
        scale  = min(lw / fw, lh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(rgb, (nw, nh))
        qi = QImage(resized.data, nw, nh, nw * 3, QImage.Format_RGB888)
        self._cam.setPixmap(QPixmap.fromImage(qi))

    def _snapshot(self):
        frame = self._pipeline.get_frame()
        if frame is None:
            return
        frame = cv2.flip(frame, 1)
        import time as _t
        path = SNAPSHOTS_DIR / f"snap_{_t.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        QMessageBox.information(self, "Saved", str(path))

    def _reset_blinks(self):
        for t in self._trackers:
            t.reset()

    def closeEvent(self, event):
        self._timer.stop()
        self._pipeline.stop()
        self._attrs.stop()
        self._faces_an.close()
        self._hand_an.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(25, 25, 25))
    pal.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Base,            QColor(18, 18, 18))
    pal.setColor(QPalette.Button,          QColor(45, 45, 45))
    pal.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,       QColor(76, 175, 80))
    pal.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()