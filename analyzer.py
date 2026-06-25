import math
import threading
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from config import LANDMARKER, GENDER_PROTO, GENDER_MODEL, AGE_PROTO, AGE_MODEL, ANALYSIS_INTERVAL_MS, EAR_THRESHOLD, EAR_CONSEC_FRAMES, HAND_LANDMARKER

_SILHOUETTE = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
_LEFT_EYE   = [33,  160, 158, 133, 153, 144]
_RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
_GENDER_LIST = ["Male", "Female"]
_AGE_LIST    = ["(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"]
_DNN_MEAN    = (78.4263377603, 87.7689143744, 114.895847746)
_FINGER_TIPS = [8, 12, 16, 20]
_FINGER_PIPS = [6, 10, 14, 18]
MESH_CONNECTIONS: list[tuple[int,int]] = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
CONTOUR_CONNECTIONS: list[tuple[int,int]] = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]

class FaceAnalyzer:
    def __init__(self):
        base = mp_python.BaseOptions(model_asset_path=str(LANDMARKER))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[dict]:
        h, w = rgb.shape[:2]
        result = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        faces = []
        for face_lm in (result.face_landmarks or []):
            pts = [(int(p.x * w), int(p.y * h), p.z) for p in face_lm]
            sil_x = [pts[i][0] for i in _SILHOUETTE if i < len(pts)]
            sil_y = [pts[i][1] for i in _SILHOUETTE if i < len(pts)]
            bx = max(0, min(sil_x) - 10)
            by = max(0, min(sil_y) - 10)
            bw = min(w - bx, max(sil_x) - min(sil_x) + 20)
            bh = min(h - by, max(sil_y) - min(sil_y) + 20)
            faces.append({
                "bbox":          (bx, by, bw, bh),
                "landmarks":     pts,
                "landmark_norm": face_lm,
            })
        return faces

    def close(self):
        self._lm.close()

class HandAnalyzer:
    def __init__(self):
        base = mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = mp_vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[dict]:
        h, w = rgb.shape[:2]
        result = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hands = []
        lm_list   = result.hand_landmarks or []
        hand_list = result.handedness     or []
        for lm_norm, handedness in zip(lm_list, hand_list):
            pts = [(int(p.x * w), int(p.y * h), p.z) for p in lm_norm]
            side = handedness[0].category_name
            count = self._count_fingers(lm_norm, side)
            hands.append({
                "landmarks":     pts,
                "landmark_norm": lm_norm,
                "handedness":    side,
                "fingers":       count,
            })
        total = sum(h["fingers"] for h in hands)
        for hand in hands:
            hand["total"] = total
        return hands

    def _count_fingers(self, lm, side: str) -> int:
        count = 0
        def dist2d(a, b):
            return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
        wrist          = lm[0]
        thumb_tip      = lm[4]
        thumb_mcp      = lm[2]
        index_mcp      = lm[5]
        finger_spread  = dist2d(lm[5], lm[17])
        tip_wrist_dist = dist2d(thumb_tip, wrist)
        mcp_wrist_dist = dist2d(thumb_mcp, wrist)
        tip_index_dist = dist2d(thumb_tip, index_mcp)
        thumb_open = (tip_wrist_dist > mcp_wrist_dist * 1.3 and tip_index_dist > finger_spread * 0.4)
        if thumb_open:
            count += 1
        for tip_i, pip_i in zip(_FINGER_TIPS, _FINGER_PIPS):
            if lm[tip_i].y < lm[pip_i].y:
                count += 1
        return count

    def close(self):
        self._lm.close()

class EyeTracker:
    def __init__(self):
        self.total_blinks = 0
        self._below = 0

    def update(self, lm_norm, w: int, h: int) -> bool:
        def ear(indices):
            pts = [(lm_norm[i].x * w, lm_norm[i].y * h) for i in indices]
            p1,p2,p3,p4,p5,p6 = pts
            return (math.dist(p2,p6) + math.dist(p3,p5)) / (2*math.dist(p1,p4) + 1e-6)
        avg = (ear(_LEFT_EYE) + ear(_RIGHT_EYE)) / 2
        blink = False
        if avg < EAR_THRESHOLD:
            self._below += 1
        else:
            if self._below >= EAR_CONSEC_FRAMES:
                self.total_blinks += 1
                blink = True
            self._below = 0
        return blink

    def reset(self):
        self.total_blinks = 0
        self._below = 0

class AttributesDNN:
    def __init__(self):
        self._lock    = threading.Lock()
        self._input   = None
        self._output  = []
        self._running = False
        self._gender_net = None
        self._age_net    = None

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="dnn")
        t.start()

    def stop(self):
        self._running = False

    def submit(self, rgb: np.ndarray, bboxes: list[tuple]):
        with self._lock:
            self._input = (rgb.copy(), list(bboxes))

    def get(self) -> list[dict]:
        with self._lock:
            return list(self._output)

    def _loop(self):
        self._gender_net = cv2.dnn.readNet(str(GENDER_PROTO), str(GENDER_MODEL))
        self._age_net    = cv2.dnn.readNet(str(AGE_PROTO),    str(AGE_MODEL))
        interval = ANALYSIS_INTERVAL_MS / 1000.0
        while self._running:
            time.sleep(interval)
            with self._lock:
                if self._input is None:
                    continue
                frame, bboxes = self._input
                self._input = None
            results = [self._infer(frame, bb) for bb in bboxes]
            with self._lock:
                self._output = results

    def _infer(self, rgb: np.ndarray, bbox: tuple) -> dict:
        bx, by, bw, bh = bbox
        h, w = rgb.shape[:2]
        m = int(min(bw, bh) * 0.2)
        crop = rgb[max(0,by-m):min(h,by+bh+m), max(0,bx-m):min(w,bx+bw+m)]
        if crop.size == 0:
            return _empty()
        bgr  = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(bgr, 1.0, (227,227), _DNN_MEAN, swapRB=False)
        self._gender_net.setInput(blob)
        gp = self._gender_net.forward()[0]
        gi = int(np.argmax(gp))
        self._age_net.setInput(blob)
        ap = self._age_net.forward()[0]
        ai = int(np.argmax(ap))
        lo, hi = _AGE_LIST[ai].strip("()").split("-")
        return {
            "gender":      _GENDER_LIST[gi],
            "gender_conf": round(float(gp[gi]) * 100, 1),
            "age":         (int(lo) + int(hi)) // 2,
            "age_range":   _AGE_LIST[ai],
        }

def _empty() -> dict:
    return {"gender": "—", "gender_conf": 0.0, "age": 0, "age_range": "—"}