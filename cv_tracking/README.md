# ForeSite AI — Person 1: Computer Vision & Tracking Module

Real-time perception and tracking layer for **ForeSite AI** (prescriptive collision prevention for construction sites).

---

## Pipeline Overview

```text
CCTV / Site Video Feed (.mp4 / RTSP / webcam)
         ↓
OpenCV Frame Ingestion (video_reader.py)
         ↓
YOLO Object Detection (detector.py)
  - Classes: Worker (person), Forklift, Excavator, Truck, Loader
  - Filters irrelevant objects (chairs, bags, signs)
         ↓
ByteTrack Multi-Object Tracking (tracker.py)
  - Persistent IDs across frames
  - Two-stage association (high-conf + low-conf matching)
         ↓
Position Extraction & Normalization (position.py)
  - Bottom-center ground contact point: x = (x1 + x2)/2, y = y2
  - Resolution-invariant normalized coordinates: [0.0, 1.0]
         ↓
Trajectory-History Buffer (trajectory_buffer.py)
  - Recent N observations (default N=12)
  - Grace period / temporary occlusion buffer (default 1.5s)
         ↓
Visualization & Structured Export (visualize.py, exporter.py)
  - Hand-off to Person 2 (output/trajectories.json)
  - Hand-off to Person 4 (output/frame_tracks.json + annotated video)
  - Debugging & training (output/trajectories.csv)
```

---

## 1. Quick Installation

```bash
# Using uv (fastest):
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard pip:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Quickstart & Run Commands

### A. Run with Built-in Demo Scenario (Zero Setup Needed)
Generates a realistic 1280x720 construction site clip with a worker and a forklift on converging trajectories, executes tracking, and exports all files:
```bash
python -m cv_tracking.run_cv_pipeline --generate-demo --save-video
```

### B. Run on a Recorded Construction Video
```bash
python -m cv_tracking.run_cv_pipeline --video path/to/construction_site.mp4 --save-video
```

### C. Run with Live Window Display
```bash
python -m cv_tracking.run_cv_pipeline --video path/to/construction_site.mp4 --gui
```

### D. Run with Custom Hyperparameters
```bash
python -m cv_tracking.run_cv_pipeline \
    --video path/to/site.mp4 \
    --conf 0.40 \
    --history-len 16 \
    --output-dir output \
    --save-video
```

---

## 3. Integration Contracts

### Person 1 → Person 2 (Trajectory Prediction & Risk)
**File**: `output/trajectories.json` (also mock file at `data/mock_trajectories.json`)

Downstream usage in `prediction`:
```python
import json

with open("output/trajectories.json") as f:
    trajectories = json.load(f)

for item in trajectories:
    track_id = item["track_id"]
    class_name = item["class_name"]
    history = item["history"]  # List of [x_norm, y_norm, timestamp_seconds]
    
    # Person 2 GRU / LSTM sequence prediction:
    future_path = predict_trajectory(history)
```

**JSON Schema**:
```json
[
  {
    "track_id": 1,
    "class_name": "worker",
    "history": [
      [0.48508, 0.51637, 3.567],
      [0.48774, 0.51512, 3.600],
      [0.49041, 0.51387, 3.633]
    ]
  },
  {
    "track_id": 2,
    "class_name": "forklift",
    "history": [
      [0.51492, 0.50145, 3.567],
      [0.51226, 0.50287, 3.600],
      [0.50959, 0.50429, 3.633]
    ]
  }
]
```

### Person 1 → Person 4 (Streamlit Dashboard & Overlay)
**File**: `output/frame_tracks.json`

Provides per-frame tracking states so the dashboard can overlay predicted paths and collision warnings:
```json
[
  {
    "frame_index": 0,
    "timestamp": 0.0,
    "tracks": [
      {
        "track_id": 1,
        "class_name": "worker",
        "bbox": [227.41, 359.10, 291.41, 467.10],
        "position": [0.20266, 0.64875],
        "timestamp": 0.0,
        "confidence": 0.94
      }
    ]
  }
]
```

### CSV Output (Training & Debugging)
**File**: `output/trajectories.csv`
```csv
timestamp,track_id,class_name,x,y
0.0,1,worker,0.20266,0.64875
0.0,2,forklift,0.79734,0.35142
0.033,1,worker,0.20453,0.64788
0.033,2,forklift,0.79547,0.35241
```

---

## 4. Module Structure

```text
cv_tracking/
├── __init__.py           # Package exports
├── config.py             # Hyperparameters, anchors, class mappings, palettes
├── video_reader.py       # OpenCV video ingestion with timestamping
├── detector.py           # YOLO detector (ONNX Runtime, Ultralytics, Synthetic fallback)
├── tracker.py            # ByteTrack multi-object tracker
├── position.py           # Bottom-center extraction & coordinate normalization
├── trajectory_buffer.py  # N-step history buffer & occlusion grace-period manager
├── visualize.py          # Bounding boxes, IDs, representative dots, motion trails
├── exporter.py           # JSON & CSV exporters + mock data generator
└── run_cv_pipeline.py    # Master CLI pipeline runner + demo clip generator

shared/
├── __init__.py
└── schemas.py            # Shared Track, TrajectoryHistory, Prediction, Risk schemas

data/
├── demo_site.mp4         # Sample generated construction site scenario
└── mock_trajectories.json# Offline mock data for Person 2 integration

output/
├── annotated_feed.mp4    # Visualized video feed
├── trajectories.json     # Person 2 input
├── frame_tracks.json     # Person 4 input
└── trajectories.csv      # CSV observation log

tests/
├── test_schemas.py       # Schema validation and roundtrip tests
├── test_position.py      # Bottom-center extraction and normalization tests
├── test_buffer.py        # Buffer length, smoothing, and expiry tests
├── test_exporter.py      # JSON/CSV exporter tests
├── test_video_reader.py  # Video ingestion tests
├── test_detector.py      # Label filtering & detector tests
├── test_tracker.py       # ByteTrack ID persistence tests
├── test_visualize.py     # HUD and overlay tests
└── test_pipeline.py      # Full end-to-end integration test
```

---

## 5. Verification & Testing

Run the test suite:
```bash
pytest tests/ -v
```
All 27 unit and end-to-end integration tests execute in under 1 second.
