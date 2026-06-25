from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / ".models"

GENDER_PROTO    = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL    = MODEL_DIR / "gender_net.caffemodel"
AGE_PROTO       = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL       = MODEL_DIR / "age_net.caffemodel"
LANDMARKER      = MODEL_DIR / "face_landmarker.task"
HAND_LANDMARKER = MODEL_DIR / "hand_landmarker.task"

CAMERA_INDEX  = 0
CAMERA_FPS    = 30
ANALYSIS_INTERVAL_MS = 250
EAR_THRESHOLD        = 0.22
EAR_CONSEC_FRAMES    = 2

BBOX_COLOR    = (0, 255, 80)
MESH_COLOR    = (80, 200, 255)
LM_COLOR      = (255, 80, 80)
HAND_COLOR    = (255, 200, 0)
BLUR_STRENGTH = 25

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)
SIDEBAR_WIDTH = 260