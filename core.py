import queue
import time
import threading
import cv2
import numpy as np
from analyzer import MESH_CONNECTIONS, CONTOUR_CONNECTIONS
from config import CAMERA_INDEX, CAMERA_FPS, BBOX_COLOR, MESH_COLOR, LM_COLOR, HAND_COLOR, BLUR_STRENGTH

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.52
_FONT_THICK = 1
_TEXT_COLOR = (255, 255, 255)
_TEXT_BG    = (20, 20, 20)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

class CameraPipeline:
    def __init__(self):
        self._q       = queue.Queue(maxsize=2)
        self._running = False
        self._cap     = None

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def get_frame(self):
        frame = None
        try:
            while True:
                frame = self._q.get_nowait()
        except queue.Empty:
            pass
        return frame

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

class Renderer:
    def draw(
        self,
        frame: np.ndarray,
        faces: list[dict],
        attrs: list[dict],
        blink_counts: list[int],
        hands: list[dict],
        flags: dict,
    ) -> np.ndarray:
        out = frame.copy()

        if flags.get("blur_bg") and faces:
            out = self._blur_background(out, faces)
        if flags.get("pixelate") and faces:
            out = self._pixelate(out, faces)

        for i, face in enumerate(faces):
            bx, by, bw, bh = face["bbox"]
            lm  = face["landmarks"]
            att = attrs[i] if i < len(attrs) else {}
            blinks = blink_counts[i] if i < len(blink_counts) else 0

            if flags.get("show_bbox"):
                self._bbox(out, bx, by, bw, bh)
            if flags.get("show_mesh"):
                self._mesh(out, lm)
            if flags.get("show_landmarks"):
                self._landmarks(out, lm)
            if flags.get("show_bbox"):
                self._label(out, bx, by, att, blinks, i)

        if hands:
            total = hands[0].get("total", 0)
            for hand in hands:
                self._hand_skeleton(out, hand["landmarks"])
            self._finger_count_badge(out, total)

        return out

    def _bbox(self, frame, bx, by, bw, bh):
        cs = min(bw, bh) // 6
        t  = 2
        for ax, ay, hx, hy, vx, vy in [
            (bx,      by,      bx+cs,    by,      bx,      by+cs),
            (bx+bw,   by,      bx+bw-cs, by,      bx+bw,   by+cs),
            (bx,      by+bh,   bx+cs,    by+bh,   bx,      by+bh-cs),
            (bx+bw,   by+bh,   bx+bw-cs, by+bh,   bx+bw,   by+bh-cs),
        ]:
            cv2.line(frame, (ax,ay), (hx,hy), BBOX_COLOR, t)
            cv2.line(frame, (ax,ay), (vx,vy), BBOX_COLOR, t)

    def _mesh(self, frame, pts):
        for (a, b) in MESH_CONNECTIONS + CONTOUR_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a][:2], pts[b][:2], MESH_COLOR, 1)

    def _landmarks(self, frame, pts):
        for p in pts:
            cv2.circle(frame, p[:2], 1, LM_COLOR, -1)

    def _label(self, frame, bx, by, att, blinks, idx):
        lines = [f"Face {idx+1}"]
        if att.get("gender") and att["gender"] != "—":
            lines.append(f"{att['gender']}  {att.get('gender_conf',0):.0f}%")
        if att.get("age"):
            lines.append(f"Age {att['age']}  {att.get('age_range','')}")
        lines.append(f"Blinks: {blinks}")
        self._text_block(frame, lines, bx, by)

    def _hand_skeleton(self, frame, pts):
        for (a, b) in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a][:2], pts[b][:2], HAND_COLOR, 2)
        for i, p in enumerate(pts):
            r = 5 if i == 0 else 3
            cv2.circle(frame, p[:2], r, HAND_COLOR, -1)
            cv2.circle(frame, p[:2], r, (0, 0, 0), 1)

    def _finger_count_badge(self, frame, total: int):
        fh, fw = frame.shape[:2]
        label  = str(total)
        size   = 3.0
        thick  = 6
        (tw, th), _ = cv2.getTextSize(label, _FONT, size, thick)
        pad = 20
        x   = fw - tw - pad * 2
        y   = pad + th

        cx, cy = x + tw // 2, y - th // 2
        r = max(tw, th) // 2 + pad
        cv2.circle(frame, (cx, cy), r, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy), r, HAND_COLOR, 3)
        cv2.putText(frame, label, (x, y), _FONT, size, HAND_COLOR, thick, cv2.LINE_AA)

        sub = "fingers"
        (sw, _), _ = cv2.getTextSize(sub, _FONT, 0.5, 1)
        cv2.putText(frame, sub, (cx - sw // 2, cy + r + 18), _FONT, 0.5, HAND_COLOR, 1, cv2.LINE_AA)

    def _text_block(self, frame, lines, bx, by):
        pad    = 4
        line_h = 17
        max_w  = max((cv2.getTextSize(l, _FONT, _FONT_SCALE, _FONT_THICK)[0][0] for l in lines), default=60)
        block_h = len(lines) * line_h + pad * 2
        fh, fw  = frame.shape[:2]
        x1 = max(0, bx)
        y1 = max(0, by - block_h - 4)
        x2 = min(fw, x1 + max_w + pad * 2)
        y2 = min(fh, y1 + block_h)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, np.full_like(roi, _TEXT_BG), 0.7, 0)
        for j, ln in enumerate(lines):
            ty = y1 + pad + (j + 1) * line_h
            cv2.putText(frame, ln, (x1 + pad, ty), _FONT, _FONT_SCALE, _TEXT_COLOR, _FONT_THICK, cv2.LINE_AA)

    def _blur_background(self, frame, faces):
        k       = BLUR_STRENGTH | 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        mask    = np.zeros(frame.shape[:2], dtype=np.uint8)
        for face in faces:
            bx, by, bw, bh = face["bbox"]
            cx, cy = bx + bw // 2, by + bh // 2
            cv2.ellipse(mask, (cx, cy), (int(bw*0.6), int(bh*0.7)), 0, 0, 360, 255, -1)
        fk = max(BLUR_STRENGTH * 2 + 1, 51)
        mask = cv2.GaussianBlur(mask, (fk, fk), 0)
        m3 = mask[:,:,np.newaxis] / 255.0
        return (frame * m3 + blurred * (1 - m3)).astype(np.uint8)

    def _pixelate(self, frame, faces):
        out = frame.copy()
        ps  = 30
        for face in faces:
            bx, by, bw, bh = face["bbox"]
            roi = out[by:by+bh, bx:bx+bw]
            if roi.size == 0:
                continue
            small = cv2.resize(roi, (max(1,bw//ps), max(1,bh//ps)), interpolation=cv2.INTER_LINEAR)
            out[by:by+bh, bx:bx+bw] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        return out