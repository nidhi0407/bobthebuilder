# ForeSite AI — Person 1 Plan
## Computer Vision & Tracking Lead

### Role objective

Build the **real-time perception layer** of ForeSite AI.

Your module must take a construction-site video or CCTV feed and convert it into a stable stream of tracked worker and machinery positions that the trajectory-prediction module can consume.

Your success criterion is:

> **Given a site video, reliably detect relevant workers and machines, preserve their identities across frames, generate clean motion histories, and hand those trajectories to Person 2 in a fixed, agreed format.**

This role owns the first three stages of the ForeSite pipeline:

```text
CCTV / Site Video
        ↓
Object Detection — YOLO
        ↓
Multi-Object Tracking — ByteTrack
        ↓
Tracked position history
        ↓
HANDOFF TO PERSON 2
Trajectory Prediction + Risk
```

The full project only works if your output is stable. If object IDs switch, trajectories jump, or timestamps are inconsistent, the downstream GRU/LSTM and risk engine will produce unreliable predictions.

---

# 1. Responsibilities

You own:

- video ingestion;
- frame preprocessing;
- object detection;
- worker and machinery class filtering;
- multi-object tracking;
- persistent track IDs;
- object position extraction;
- trajectory-history buffers;
- coordinate normalization;
- per-frame structured outputs;
- CV-side visualization for debugging;
- export of tracked trajectories for Person 2;
- integration support with the final Streamlit dashboard.

You do **not** own:

- GRU/LSTM training;
- future-trajectory prediction;
- collision-risk scoring;
- counterfactual intervention simulation;
- final intervention ranking;
- full dashboard logic.

Your job is to produce the cleanest possible input for those modules.

---

# 2. End-to-end responsibility

```text
INPUT
Recorded video / CCTV stream

        ↓

FRAME INGESTION
OpenCV reads frames and timestamps them

        ↓

YOLO OBJECT DETECTION
Detect:
- workers / persons
- forklifts
- excavators
- trucks
- loaders
- any relevant equipment in the chosen demo

        ↓

CLASS FILTERING
Discard irrelevant detections

        ↓

BYTETRACK
Associate detections over time
Assign stable track IDs

        ↓

POSITION EXTRACTION
Convert bounding boxes into representative ground/image positions

        ↓

TRAJECTORY BUFFER
Maintain recent (x, y, t) history for every track

        ↓

NORMALIZATION
Convert pixel positions into normalized coordinates

        ↓

STRUCTURED OUTPUT
Send tracks + history to Person 2

        ↓

DOWNSTREAM
GRU/LSTM → risk engine → counterfactual engine → dashboard
```

---

# 3. Integration contract with Person 2

This must be agreed on at the very beginning.

Person 2 should **not** depend on YOLO-specific objects, Ultralytics result classes, or ByteTrack internals.

Your module should expose plain Python dictionaries / JSON-compatible structures.

## Recommended per-frame track object

```python
{
    "track_id": 4,
    "class_name": "worker",
    "bbox": [x1, y1, x2, y2],
    "position": [0.421, 0.673],
    "timestamp": 2.35,
    "confidence": 0.91
}
```

Where:

- `track_id` = stable ByteTrack ID;
- `class_name` = normalized label such as `worker`, `forklift`, `excavator`;
- `bbox` = original pixel bounding box;
- `position` = normalized x,y point;
- `timestamp` = seconds from video start;
- `confidence` = detector confidence.

## Recommended trajectory-history object

```python
{
    "track_id": 4,
    "class_name": "worker",
    "history": [
        [0.381, 0.702, 1.85],
        [0.392, 0.694, 1.95],
        [0.405, 0.684, 2.05],
        [0.421, 0.673, 2.15]
    ]
}
```

Person 2 should be able to call:

```python
future_path = predict_trajectory(track["history"])
```

without knowing anything about detection or tracking.

---

# 4. Coordinate convention

Use a single, consistent convention.

## Recommended point

For workers:

```text
bottom-center of bounding box
```

Approximation:

```python
x = (x1 + x2) / 2
y = y2
```

This is usually preferable to the geometric center because it better approximates the person's contact point with the ground in image coordinates.

For machines:

- center or bottom-center is acceptable;
- whichever is chosen, keep it consistent.

## Recommended normalization

```python
x_norm = x / frame_width
y_norm = y / frame_height
```

Therefore all coordinates lie approximately in:

```text
0 ≤ x ≤ 1
0 ≤ y ≤ 1
```

This prevents Person 2's model from becoming tied to one video resolution.

---

# 5. File / module structure

Recommended structure:

```text
foresite/
│
├── cv_tracking/
│   ├── __init__.py
│   ├── config.py
│   ├── video_reader.py
│   ├── detector.py
│   ├── tracker.py
│   ├── position.py
│   ├── trajectory_buffer.py
│   ├── visualize.py
│   ├── exporter.py
│   └── run_cv_pipeline.py
│
├── prediction/
├── risk/
├── intervention/
├── dashboard/
└── shared/
    └── schemas.py
```

Use `shared/schemas.py` or a shared JSON schema so all team members use the same field names.

---

# 6. Detailed implementation plan

## Phase A — Video ingestion

### Goal

Read a test construction-site video reliably.

### Tasks

- install OpenCV;
- open `.mp4` or selected video format;
- read FPS;
- extract frame dimensions;
- assign timestamps;
- display frames;
- confirm no dropped/invalid frames.

### Output

```python
frame
frame_index
timestamp
frame_width
frame_height
```

### Completion check

You can run:

```bash
python run_cv_pipeline.py --video demo.mp4
```

and the video plays correctly.

---

# 7. Phase B — YOLO detection

### Goal

Detect the worker and relevant moving machinery.

### Initial strategy

Start with a pretrained Ultralytics YOLO model.

Do **not** begin by training a custom detector unless necessary.

### Minimum useful classes

For the hackathon MVP:

- person / worker;
- one main machine class used in the demo.

For example:

```text
worker
forklift
```

That is sufficient to demonstrate the entire ForeSite concept.

### Secondary classes if reliable

- excavator;
- truck;
- loader;
- crane.

### Detection output

```python
[
    {
        "class_name": "worker",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.94
    },
    {
        "class_name": "forklift",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.88
    }
]
```

### Important rule

Do not waste time detecting every object.

Ignore:

- chairs;
- traffic signs unless relevant;
- tools;
- irrelevant vehicles;
- background objects.

Only pass objects important to worker-machine safety.

---

# 8. Phase C — Construction-specific detection fallback

If generic YOLO cannot detect the machinery reliably:

### Fallback order

```text
1. Try pretrained YOLO
        ↓
2. Try a construction-specific pretrained checkpoint
        ↓
3. Switch to a clearer demo video
        ↓
4. Fine-tune only if a ready dataset + weights are available
```

Do not spend several hackathon hours building a custom machinery detector from scratch.

The ForeSite innovation is downstream prediction and intervention, so your priority is a reliable data stream.

---

# 9. Phase D — ByteTrack integration

### Goal

Maintain stable IDs across frames.

YOLO alone produces detections independently.

ByteTrack should turn:

```text
Frame 1: person
Frame 2: person
Frame 3: person
```

into:

```text
Frame 1: Worker #4
Frame 2: Worker #4
Frame 3: Worker #4
```

Similarly:

```text
Forklift #2
```

must remain `#2` while visible.

### Completion check

Overlay:

```text
Worker #4
Forklift #2
```

on the video.

Run a clip and visually verify that:

- IDs do not constantly change;
- worker is not confused with machinery;
- tracks survive brief detector misses;
- trajectories are reasonably smooth.

---

# 10. Phase E — Trajectory buffer

Maintain recent motion history for every track.

Example:

```python
trajectory_store = {
    4: {
        "class_name": "worker",
        "history": [
            [0.31, 0.65, 2.1],
            [0.32, 0.64, 2.2],
            [0.34, 0.63, 2.3]
        ]
    }
}
```

### Recommended buffer size

Initially:

```text
10–20 recent observations
```

Person 2 can later specify the exact sequence length required by the GRU/LSTM.

Your code should make this configurable:

```python
HISTORY_LENGTH = 12
```

---

# 11. Phase F — Visualization

Draw historical trails behind objects.

Example:

```text
• • • • • 👷
```

and:

```text
■ ■ ■ ■ 🚜
```

This helps with:

- debugging;
- ID-switch detection;
- detecting noisy coordinates;
- pitch/demo clarity.

Your overlay should ideally show:

```text
Worker #4
Forklift #2
```

plus recent motion trails.

Future predicted trajectories will be added later by Person 4 using Person 2's output.

---

# 12. Phase G — Export interface

Create an exporter that can output:

## JSON

```json
{
  "timestamp": 4.2,
  "tracks": [
    {
      "track_id": 4,
      "class_name": "worker",
      "position": [0.42, 0.67]
    },
    {
      "track_id": 2,
      "class_name": "forklift",
      "position": [0.71, 0.35]
    }
  ]
}
```

## Optional CSV for debugging/training

```text
timestamp,track_id,class_name,x,y
0.10,4,worker,0.31,0.65
0.20,4,worker,0.32,0.64
0.30,4,worker,0.34,0.63
```

This lets Person 2 train/test independently.

---

# 13. Critical integration rules

## Rule 1 — fixed labels

Agree on labels such as:

```text
worker
forklift
excavator
truck
loader
```

Do not send one frame as `person` and another as `worker`.

Normalize the class names inside your module.

---

## Rule 2 — stable coordinate format

Always use:

```text
[x_norm, y_norm]
```

for downstream trajectory prediction.

Do not sometimes send pixels and sometimes normalized values.

---

## Rule 3 — stable time units

Use:

```text
seconds
```

for timestamps.

---

## Rule 4 — missing-frame behavior

If an object temporarily disappears:

- do not instantly destroy its trajectory history;
- allow a short timeout;
- preserve history in case ByteTrack recovers the ID.

Example:

```text
TRACK_EXPIRY = 1–2 seconds
```

This also supports a future Ghost Worker extension.

---

# 14. How your module connects to the rest of ForeSite

## Person 1 → Person 2

You provide:

```text
Stable tracked trajectories
```

Person 2 consumes:

```text
last N positions
```

and outputs:

```text
future predicted positions
risk
time-to-conflict
collision point
```

---

## Person 2 → Person 3

Person 3 takes the predicted future and simulates interventions.

Example:

```text
machine speed -20%
machine speed -40%
worker stops
worker reroutes
```

---

## Person 3 → Person 4

Person 3 outputs:

```python
{
    "action": "reduce_vehicle_speed",
    "magnitude": 0.38,
    "time_window": 1.7,
    "resulting_risk": 0.08
}
```

---

## Person 1 → Person 4 directly

Your module also provides Person 4 with:

- current frame;
- bounding boxes;
- track IDs;
- historical motion trails;
- class labels.

Person 4 overlays Person 2's future paths and Person 3's recommended intervention on top.

---

# 15. Shared schema meeting — first hour

All four members should agree on the following before developing independently.

```python
Track = {
    "track_id": int,
    "class_name": str,
    "bbox": list,
    "position": list,
    "timestamp": float,
    "confidence": float
}
```

```python
Prediction = {
    "track_id": int,
    "future_positions": list
}
```

```python
RiskResult = {
    "worker_id": int,
    "machine_id": int,
    "risk_score": float,
    "time_to_conflict": float,
    "minimum_distance": float,
    "collision_point": list
}
```

```python
Recommendation = {
    "action": str,
    "magnitude": float,
    "time_window": float,
    "resulting_risk": float
}
```

Freeze these field names early.

---

# 16. Mock data so Person 2 is never blocked

Before your real CV pipeline is finished, generate a mock file such as:

```json
[
  {
    "track_id": 1,
    "class_name": "worker",
    "history": [
      [0.20, 0.60, 0.0],
      [0.22, 0.59, 0.1],
      [0.24, 0.58, 0.2]
    ]
  },
  {
    "track_id": 2,
    "class_name": "forklift",
    "history": [
      [0.80, 0.30, 0.0],
      [0.77, 0.33, 0.1],
      [0.74, 0.36, 0.2]
    ]
  }
]
```

Send this to Person 2 immediately.

This allows all team members to develop in parallel.

---

# 17. 30-hour schedule for Person 1

## Hour 0–1

- agree shared schemas;
- choose demo video;
- set up branch;
- confirm worker + machine target classes.

## Hour 1–3

- OpenCV video ingestion;
- frame timestamps;
- test pipeline.

## Hour 3–6

- YOLO integration;
- worker detection;
- machinery detection;
- class filtering.

## Hour 6–9

- ByteTrack integration;
- stable IDs;
- visual ID validation.

## Hour 9–11

- coordinate extraction;
- normalization;
- trajectory buffers.

## Hour 11–12

- hand real track histories to Person 2;
- confirm model input compatibility.

### By Hour 12 your core responsibility should work.

## Hour 12–15

- improve tracking stability;
- fix ID switches;
- test second scenario.

## Hour 15–18

- integration with Person 4;
- expose annotated frames + tracks;
- test end-to-end data path.

## Hour 18–22

- optimize latency;
- adjust detection thresholds;
- handle short occlusions.

## Hour 22–25

Optional only:

- Ghost Worker behavior;
- better construction-specific model;
- perspective calibration.

## Hour 25–27

- full demo test;
- record backup video;
- freeze core CV code.

## Hour 27–30

- support integration;
- rehearse technical explanation;
- fix only critical bugs.

---

# 18. MVP definition for Person 1

Your module is complete when:

- a site video loads;
- at least one worker is reliably detected;
- at least one relevant machine is reliably detected;
- each receives a persistent ID;
- positions are normalized;
- a short history is stored;
- history is exported in the agreed schema;
- Person 2 can successfully consume it;
- Person 4 can draw your current boxes and track IDs.

That is enough.

---

# 19. Nice-to-have features

Only after the MVP works:

- construction-specific custom detector;
- multiple machine classes;
- perspective transformation;
- ground-plane projection;
- Ghost Worker;
- confidence decay during occlusion;
- dynamic tracking thresholds;
- multiple-camera support.

Do not allow these to block integration.

---

# 20. Failure modes and fallback plan

## Problem: machinery is not detected

Fallback:

```text
use construction-specific weights
        ↓
or choose a cleaner demo clip
```

---

## Problem: ByteTrack ID switches

Try:

- detector confidence tuning;
- larger tracking buffer;
- cleaner video;
- fewer irrelevant classes;
- slower test clip for demo.

---

## Problem: trajectory is jittery

Apply light smoothing to positions.

For example:

```text
moving average / exponential smoothing
```

Do not heavily smooth and introduce large lag.

---

## Problem: live pipeline becomes too slow

Fallback:

- lower video resolution;
- process every second frame;
- use smaller YOLO model;
- use prerecorded video;
- precompute tracks for final fallback demo.

---

# 21. Testing checklist

Before handing off to Person 2:

- [ ] Track IDs remain stable for the demo clip.
- [ ] Class names are normalized.
- [ ] Positions are normalized.
- [ ] Timestamps are in seconds.
- [ ] No NaN/invalid coordinates.
- [ ] Histories contain chronological observations.
- [ ] Missing tracks expire gracefully.
- [ ] JSON/CSV export works.
- [ ] Person 2 can load the exported sample.
- [ ] Person 4 can display current tracks.

Before final demo:

- [ ] Detection works on chosen demo video.
- [ ] No obvious ID switching during key collision sequence.
- [ ] Worker and machine trails look visually smooth.
- [ ] Inference is fast enough for a convincing demo.
- [ ] Backup prerecorded output exists.

---

# 22. What you say in the presentation

### Short answer

> **I built ForeSite's perception and tracking layer. I integrated YOLO-based worker and heavy-equipment detection with ByteTrack multi-object tracking, preserved each object's identity across video frames, and transformed raw detections into normalized movement histories that feed our trajectory-prediction model.**

### If asked why tracking is necessary

> **Detection only tells us where a worker or machine is in one frame. To predict an accident, we need to understand how that same worker or machine has been moving over time. Multi-object tracking provides that temporal history.**

### If asked what your module outputs

> **For every tracked object we output a stable ID, object class, bounding box, normalized position, timestamp, and short movement history. The prediction model consumes that history to forecast future motion.**

---

# 23. Final goal

Do not judge your success by how many construction objects YOLO detects.

Judge it by this question:

> **Can Person 2 reliably receive a clean trajectory for the same worker and machine over time?**

If yes, your part of ForeSite is doing exactly what the rest of the system needs.
