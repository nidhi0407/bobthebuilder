# 🚧 Bobthebuilder

### **Predict danger before it happens. Prevent it before it becomes an accident.**

ForeSite AI is a computer-vision safety system for construction sites that uses existing CCTV footage to **predict potential worker–machinery collisions seconds before they happen**.

Instead of simply detecting hazards that already exist, ForeSite analyzes the movement of workers and heavy machinery, forecasts their future trajectories, estimates collision risk, and determines what intervention could prevent the incident.

> **From safety monitoring to accident prevention.**

---

## 💡 The Problem

Construction sites are dynamic environments where workers, vehicles, and heavy machinery constantly move through shared spaces.

Most computer-vision safety systems focus on detecting what is happening **right now**:

* Is a worker wearing a helmet?
* Is a worker too close to a machine?
* Is someone inside a restricted area?
* Has someone fallen?

But distance alone doesn't tell us whether a collision is actually going to happen.

Consider this:

```text
Worker  ─────────────────────►

                         X  ← future collision

Machine ────────────────────►
```

The worker and machine may currently be several metres apart and therefore appear safe.

But if their trajectories intersect in 3 seconds, the situation is already dangerous.

### ForeSite asks a different question:

> **"What is likely to happen next, and what can we do about it?"**

---

## 🎯 What ForeSite Does

ForeSite follows a five-stage pipeline:

```text
       CCTV VIDEO
           │
           ▼
    ┌──────────────┐
    │ Object       │
    │ Detection    │
    │    YOLO      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Object       │
    │ Tracking     │
    │  ByteTrack   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Trajectory    │
    │ Prediction    │
    │  GRU / LSTM   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Risk Engine   │
    │               │
    │ TTC           │
    │ Distance      │
    │ Velocity      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Counterfactual│
    │ Simulation    │
    └──────┬───────┘
           │
           ▼
    🚨 ACTIONABLE
       WARNING
```

---

# 🔍 How It Works

## 1. Detect

ForeSite processes a CCTV feed and detects relevant objects such as:

* 👷 Workers
* 🚜 Heavy machinery
* 🚛 Vehicles
* 🏗️ Construction equipment

Each detected object is assigned a bounding box and tracked over time.

---

## 2. Track

Detection alone only tells us where an object is.

Tracking tells us **where the same object has been moving**.

For example:

```text
Worker #17

t-4   ●
t-3    ●
t-2     ●
t-1      ●
t       ●
```

ForeSite maintains a short history of object positions to understand movement patterns.

---

## 3. Predict

The recent movement history is passed to a lightweight GRU/LSTM trajectory model.

### Input

```text
(x₁, y₁)
(x₂, y₂)
(x₃, y₃)
...
(x₈, y₈)
```

### Output

```text
future position 1
future position 2
future position 3
...
future position 10
```

This gives ForeSite an estimate of where the worker or machine is likely to be several seconds into the future.

---

# ⚠️ 4. Predict Collision Risk

ForeSite compares the predicted trajectories of workers and machinery.

Instead of asking:

> "Are they close?"

it asks:

> **"Will their future paths become dangerously close?"**

The risk engine considers factors such as:

* Predicted minimum separation
* Time to conflict
* Relative velocity
* Direction of movement
* Current position
* Potential blind spots

Example:

```text
CURRENT STATE

Worker ↔ Excavator
Distance: 6.8 m

              ↓

PREDICTED STATE

Minimum separation: 0.42 m
Time to conflict: 3.4 sec

              ↓

RISK

🔴 91% — CRITICAL
```

---

# 🧠 5. Counterfactual Prevention

This is the core idea behind ForeSite.

Once a potential collision is predicted, the system doesn't stop at:

> 🚨 "Danger!"

Instead, it asks:

> **"What could we change right now to prevent it?"**

ForeSite simulates possible interventions.

```text
Scenario                         Predicted Risk

No intervention                       91%

Worker stops                         14%

Machine slows 25%                    64%

Machine slows 50%                     8%

Worker changes direction             21%
```

The system identifies the smallest intervention that brings the predicted risk below a safe threshold.

### Example output

```text
🚨 COLLISION PREDICTED

Worker #17
Excavator #3

Time to conflict: 3.4 sec
Risk: 91%

RECOMMENDED ACTION

🛑 Reduce excavator speed by ≥50%
```

This turns ForeSite from a **hazard detector** into a **decision-support system**.

---

# 👻 Ghost Worker

Construction sites have another major problem:

## Occlusion.

A worker can temporarily disappear behind a vehicle or piece of machinery.

A conventional detection system may interpret this as:

> "Worker no longer exists."

ForeSite maintains the worker's track and predicts their position during short periods of occlusion.

```text
          CAMERA VIEW

👷 ───────────────►

████████████████████
████   EXCAVATOR ████
████████████████████

       👻 - - - - - ►
       predicted worker
```

The worker isn't treated as actually visible.

Instead, the system explicitly represents the location as **predicted/uncertain**.

This prevents a temporary loss of visual detection from automatically eliminating a potentially dangerous interaction.

---

# 👤 Who Uses ForeSite?

ForeSite is designed primarily for **site supervisors and safety officers**, rather than requiring individual workers to interact with the system.

A supervisor can use the dashboard to:

### In real time

Receive warnings about imminent collisions.

```text
🚨 Zone B

Worker #17
+
Excavator #3

Collision predicted in 3.4 sec.

Recommended:
Stop excavator movement.
```

### After an incident

Review near-misses and understand what happened.

```text
TODAY'S NEAR MISSES

08:42  Worker ↔ Truck       HIGH
10:13  Worker ↔ Excavator   HIGH
11:47  Worker ↔ Crane       MEDIUM
```

### Over time

Identify recurring hazards.

```text
SITE SAFETY SUMMARY

142 potential conflicts
23 high-risk events
7 recurring locations

Highest-risk zone:
Zone B

Most common issue:
Workers entering vehicle
operating areas.
```

This allows the safety team to move from **reacting to incidents** toward **changing site conditions that repeatedly create them**.

---

# 🖥️ Dashboard

The ForeSite dashboard provides a real-time view of:

* CCTV feed
* Detected workers and machinery
* Current trajectories
* Predicted trajectories
* Collision points
* Time-to-conflict
* Risk score
* Recommended intervention
* Recent near-miss events
* Top-down site visualization

Example:

```text
┌──────────────────────────────────────────────────────┐
│                  FORESITE AI                         │
│              ● LIVE SAFETY MONITOR                  │
├────────────────────────┬─────────────────────────────┤
│                        │                             │
│       CCTV FEED        │       🔴 CRITICAL           │
│                        │                             │
│   👷 - - - - - - X     │   Collision in 3.4 sec     │
│                  \     │   Risk: 91%                │
│                   🚜   │                             │
│                        │   Worker #17                │
│                        │   Excavator #3              │
├────────────────────────┴─────────────────────────────┤
│                  INTERVENTION ENGINE                  │
│                                                      │
│ No intervention                 91%                  │
│ Worker stops                    14%                  │
│ Vehicle -25%                    64%                  │
│ Vehicle -50%                     8%                  │
│                                                      │
│ 🚨 RECOMMENDED: Reduce vehicle speed ≥50%            │
└──────────────────────────────────────────────────────┘
```

---

# 🏗️ Key Innovation

Existing construction-safety systems can already perform tasks such as:

* PPE detection
* Worker detection
* Machinery detection
* Proximity monitoring
* Unsafe-action recognition
* Fall detection

ForeSite focuses on the step beyond detection:

```text
Existing approach

DETECT
  ↓
ALERT


ForeSite

DETECT
  ↓
TRACK
  ↓
PREDICT
  ↓
SIMULATE
  ↓
RECOMMEND
  ↓
PREVENT
```

The central innovation is the combination of:

### **Future trajectory prediction + counterfactual intervention**

Rather than simply reporting that a hazard exists, ForeSite estimates **what will happen next and which intervention can prevent it**.

---

# 🧪 Example Scenario

A worker is walking across a construction site while an excavator reverses nearby.

### At first:

```text
Distance: 8.1 m
Risk: 4%

🟢 SAFE
```

A few moments later:

```text
Distance: 6.8 m
Predicted minimum separation: 0.42 m
Time to conflict: 3.4 sec

🔴 CRITICAL
Risk: 91%
```

ForeSite runs possible interventions:

```text
Worker stops       → 14%
Vehicle -25%       → 64%
Vehicle -50%       → 8%
Worker reroutes    → 21%
```

The system recommends:

> **Reduce vehicle speed by ≥50%.**

The supervisor can then communicate the intervention to the equipment operator.

---

# 🛠️ Tech Stack

| Component             | Technology         |
| --------------------- | ------------------ |
| Object Detection      | YOLO / Ultralytics |
| Object Tracking       | ByteTrack          |
| Trajectory Prediction | GRU / LSTM         |
| Computer Vision       | OpenCV             |
| Numerical Processing  | NumPy              |
| ML Framework          | PyTorch            |
| Dashboard             | Streamlit          |
| Visualization         | Plotly / OpenCV    |
| Language              | Python             |

---

# 📁 Project Structure

```text
foresite-ai/
│
├── models/
│   ├── detector/
│   └── trajectory/
│
├── detection/
│   └── detector.py
│
├── tracking/
│   └── tracker.py
│
├── prediction/
│   └── trajectory_model.py
│
├── risk/
│   ├── collision.py
│   ├── risk_engine.py
│   └── counterfactual.py
│
├── visualization/
│   ├── trajectories.py
│   └── birdseye.py
│
├── dashboard/
│   └── app.py
│
├── data/
│
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Clone the repository

```bash
git clone <repository-url>
cd foresite-ai
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Then open the local Streamlit URL displayed in your terminal.

---

# 📊 Evaluation Metrics

ForeSite can be evaluated using metrics that actually matter for preventive safety:

### Prediction

* Trajectory prediction error
* Collision prediction accuracy
* False positive rate

### Safety

* Time-to-warning
* Minimum predicted separation
* Risk reduction after intervention

### System

* Detection FPS
* End-to-end latency
* Tracking stability

A particularly important metric is:

> **Warning lead time — how many seconds before the predicted conflict does ForeSite issue a warning?**

A system that predicts a collision **0.2 seconds before impact** is much less useful than one that gives a supervisor **3–5 seconds to intervene**.

---

# 🔮 Future Work

Potential extensions include:

* Multi-camera tracking
* 3D site reconstruction
* Integration with construction-site maps/BIM
* More sophisticated trajectory models
* Crane/load trajectory prediction
* Fall-risk prediction
* Dynamic exclusion-zone generation
* Mobile alerts for supervisors
* Historical safety analytics
* Integration with site access/control systems

---

# ⚠️ Limitations

ForeSite is a prototype and should be treated as a **decision-support system**, not an autonomous safety controller.

Predictions can be affected by:

* Poor camera placement
* Occlusion
* Low video quality
* Unusual worker movement
* Sudden machinery movements
* Perspective distortion
* Insufficient training data

The system should therefore support, rather than replace, human safety procedures and professional judgment.

---

# 🌍 Impact

Construction accidents can happen in seconds.

Traditional safety systems often focus on identifying hazards after they become visible.

ForeSite aims to shift the focus toward **anticipation**.

```text
                  BEFORE

             Hazard appears
                   ↓
                Detect
                   ↓
                Alert
                   ↓
             Human reacts


                  FORESITE

             Observe movement
                   ↓
            Predict future
                   ↓
          Identify potential
              conflict
                   ↓
          Simulate interventions
                   ↓
         Recommend prevention
                   ↓
             Human acts
```

### **See the danger before it happens.**

### **Know what to do about it.**

---

## 🏆 Hackathon Pitch

> **Most construction safety AI detects danger when it is already happening. ForeSite predicts it before it happens.**
>
> We use existing CCTV to track workers and heavy machinery, predict their trajectories seconds into the future, identify potential collisions, and simulate possible interventions to determine what action could prevent the accident.
>
> **ForeSite turns CCTV from a surveillance system into a predictive safety copilot.**

---

## 📌 One-Line Description

**ForeSite AI is a predictive construction-safety system that forecasts worker–machinery collisions and recommends interventions before accidents happen.**
