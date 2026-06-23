# Camshot

Real-time camera analysis using MediaPipe and OpenCV, built with PySide6.

## Features

- **Face detection** — up to 4 faces simultaneously with bounding box overlay
- **Gender & age** — estimated via OpenCV Caffe DNN models (runs on background thread)
- **Blink counter** — Eye Aspect Ratio (EAR) algorithm, per face
- **Finger counting** — detects up to 2 hands, counts extended fingers (0–10 total)
- **Face mesh** — 478-point landmark tessellation overlay
- **Background blur** — feathered ellipse mask keeps faces sharp
- **Pixelate** — privacy mode, pixelates detected face regions

## Project structure

```
Camshot/
├── app.py         # PySide6 main window + sidebar
├── core.py        # Camera pipeline + frame renderer
├── analyzer.py    # FaceAnalyzer · HandAnalyzer · EyeTracker · AttributesDNN
├── config.py      # Paths, thresholds, colors
├── snapshots/     # Saved snapshots (auto-created)
└── .models/       # ML model files (see Setup)
    ├── face_landmarker.task
    ├── hand_landmarker.task
    ├── gender_deploy.prototxt
    ├── gender_net.caffemodel
    ├── age_deploy.prototxt
    └── age_net.caffemodel
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python app.py
```

## How it works

### Threading model

| Thread | Responsibility |
|--------|---------------|
| Main (Qt) | UI rendering, sidebar updates |
| Camera thread | OpenCV capture → frame queue (daemon) |
| DNN thread | Gender/age inference every 250 ms (daemon) |

Face mesh, blink, and hand analysis run on the main thread per frame — they're fast enough (~5–10 ms each with MediaPipe).

### Finger counting

Four fingers (index → pinky): tip landmark Y < PIP landmark Y → finger is extended.

Thumb: uses distance-based approach instead of X-axis comparison, which breaks when the hand is rotated. If `dist(tip→wrist) > dist(MCP→wrist) × 1.3` and the tip is not tucked behind the index finger, the thumb is counted as open. This works for both palm-facing and back-of-hand orientations.

### Blink detection

Eye Aspect Ratio (EAR) computed from 6 MediaPipe landmarks per eye:

```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```

A blink is registered when EAR drops below `0.22` for at least 2 consecutive frames.

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Camera device index |
| `ANALYSIS_INTERVAL_MS` | `250` | Gender/age inference interval |
| `EAR_THRESHOLD` | `0.22` | Blink detection sensitivity |
| `EAR_CONSEC_FRAMES` | `2` | Frames below threshold to count as blink |
| `BLUR_STRENGTH` | `25` | Background blur kernel size |

## Snapshots

Press **Snapshot** in the sidebar to save the current frame as a JPEG to the `snapshots/` folder.

## Notes

- Gender and age estimates are probabilistic and may not be accurate for all faces
- Finger counting works best with a plain background and good lighting
