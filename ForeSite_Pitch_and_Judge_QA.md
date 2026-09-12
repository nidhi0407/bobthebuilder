# ForeSite AI — Pitch, Review & Judge Q&A Guide

## Project name

# ForeSite AI
### Predictive and Prescriptive Construction Safety

---

# 1. One-line pitch

> **ForeSite AI predicts worker–machine collisions before they occur and determines the minimum action required to prevent the predicted accident.**

---

# 2. Short project description

ForeSite AI is an AI-powered construction-safety system that uses existing CCTV or site video to detect and track workers and heavy machinery, predict where they are likely to move over the next few seconds, identify future collision risks, and recommend an immediate preventive action. When a dangerous interaction is predicted, ForeSite goes beyond a conventional warning by simulating multiple possible interventions — such as slowing the machine, stopping it, pausing the worker, or rerouting movement — and selecting the least disruptive action that reduces the predicted risk below a safe threshold.

---

# 3. The problem

Construction sites are dynamic environments where:

- workers;
- forklifts;
- excavators;
- loaders;
- trucks;
- cranes;
- temporary structures;
- blind spots;

all operate in close proximity.

A dangerous interaction can develop in seconds.

Current safety approaches often detect:

- PPE violations;
- restricted-zone entry;
- current proximity;
- unsafe behavior;
- existing hazards.

The weakness is that many systems react when danger is already present.

A worker and forklift may still be several metres apart while already moving toward the same future collision point.

Therefore the real problem is not only:

> **Where are they now?**

It is:

> **Where will they be a few seconds from now, and what can we do right now to prevent the collision?**

---

# 4. Why ForeSite is needed

A conventional warning system can tell a worker or operator:

```text
DANGER
```

But the human must still:

1. understand what is happening;
2. judge how serious it is;
3. decide what action to take;
4. determine how urgently to act;
5. respond before the collision.

ForeSite attempts to shorten this decision loop.

Instead of:

```text
Detect danger
     ↓
Alert
     ↓
Human decides what to do
```

ForeSite aims for:

```text
Detect
     ↓
Predict
     ↓
Evaluate possible interventions
     ↓
Recommend the safest minimal action
```

---

# 5. How ForeSite works

```text
CCTV / SITE VIDEO
        ↓
YOLO OBJECT DETECTION
        ↓
BYTETRACK OBJECT TRACKING
        ↓
RECENT MOTION HISTORY
        ↓
GRU / LSTM TRAJECTORY PREDICTION
        ↓
FUTURE COLLISION-RISK ENGINE
        ↓
COUNTERFACTUAL SIMULATION
        ↓
INTERVENTION RANKING
        ↓
MINIMUM SAFE ACTION
        ↓
STREAMLIT DASHBOARD
```

---

# 6. Step-by-step example

Imagine:

```text
Worker                        Forklift

   👷                           🚜

Current distance = 6.2 m
```

A normal proximity alarm may still show:

```text
SAFE
```

ForeSite tracks both objects and predicts their next few seconds of motion.

```text
Worker future path
       \
        \
         X  ← predicted conflict
        /
       /
Forklift future path
```

ForeSite calculates:

```text
Current distance:              6.2 m
Predicted minimum separation:  0.38 m
Time to conflict:              3.4 s
Risk:                          CRITICAL
```

Now instead of only generating an alert, ForeSite simulates alternative futures.

```text
No action           → 94% risk
Machine -10% speed  → 81%
Machine -20% speed  → 59%
Machine -30% speed  → 28%
Machine -40% speed  →  7% SAFE
Machine stop        →  1%
Worker stop         →  9%
```

ForeSite therefore outputs:

> **Reduce forklift speed by at least 40% within the available response window.**

---

# 7. Core innovation

The project should **not** claim that object detection, trajectory prediction, or collision warnings are completely new.

Those ideas already exist independently.

The innovation is the transition from:

```text
DETECTION
"What is happening?"
```

to:

```text
PREDICTION
"What will happen?"
```

to:

```text
PRESCRIPTION
"What should we do now so it does not happen?"
```

ForeSite combines:

- computer vision;
- multi-object tracking;
- trajectory forecasting;
- collision-risk prediction;
- counterfactual simulation;
- minimum-intervention optimization.

The differentiator is the **prescriptive decision layer**.

---

# 8. How ForeSite differs from a proximity alarm

## Proximity alarm

```text
Worker enters fixed danger radius
        ↓
Alert
```

It is mainly based on the worker's current location relative to a machine.

## ForeSite

```text
Worker and machine currently several metres apart
        ↓
Predict both future trajectories
        ↓
Forecast dangerous interaction
        ↓
Evaluate multiple interventions
        ↓
Recommend minimum safe action
```

ForeSite therefore focuses on **future interaction**, not only present distance.

---

# 9. How ForeSite differs from a normal collision-warning system

A predictive collision-warning system may say:

> **Collision likely in 3 seconds.**

ForeSite attempts to add:

> **Collision likely in 3 seconds. Slowing the machine by approximately 40% is the smallest tested intervention that moves the predicted situation below the safety threshold.**

That extra step changes the output from:

```text
warning
```

to:

```text
decision support
```

---

# 10. Why minimum intervention matters

The system should not always recommend:

```text
STOP EVERYTHING
```

because that could create unnecessary operational disruption and alarm fatigue.

ForeSite tries to identify the smallest safe response.

Example:

```text
20% slowdown → unsafe
30% slowdown → unsafe
40% slowdown → safe
100% stop    → safe but unnecessarily disruptive
```

Therefore ForeSite recommends:

```text
40% slowdown
```

rather than a full stop.

The long-term objective is to balance:

```text
worker safety
+
minimum operational disruption
```

Safety remains the constraint; efficiency is optimized only among safe options.

---

# 11. Why construction?

Construction is a strong application because:

- workers and heavy machinery share rapidly changing spaces;
- site layouts change frequently;
- blind spots are common;
- machines can cause severe struck-by injuries;
- safety supervisors cannot continuously watch every worker–machine interaction;
- many sites already have cameras.

ForeSite is designed as an intelligence layer on top of existing video infrastructure rather than requiring an entirely new sensing system for the prototype.

---

# 12. Where ForeSite is useful

Potential environments include:

- active construction sites;
- road construction;
- warehouses attached to construction operations;
- excavation zones;
- material-handling areas;
- loading/unloading zones;
- forklift-heavy work areas;
- industrial project sites;
- infrastructure construction.

Possible monitored interactions include:

```text
worker ↔ forklift
worker ↔ excavator
worker ↔ truck
worker ↔ loader
worker ↔ crane operating zone
```

The first hackathon prototype should focus on one clear worker–machine interaction.

---

# 13. AI/ML components

ForeSite contains real ML components.

## Model 1 — YOLO

Purpose:

```text
detect workers and machinery
```

## Tracking — ByteTrack

Purpose:

```text
maintain persistent object identity
```

## Model 2 — GRU or LSTM

Purpose:

```text
learn temporal movement patterns
and predict future coordinates
```

## Mathematical risk engine

Purpose:

- minimum future separation;
- time-to-conflict;
- relative velocity;
- path intersection;
- risk score.

## Counterfactual engine

Purpose:

```text
simulate alternative actions
```

No additional neural network is necessary for this layer.

---

# 14. Why use ML for trajectory prediction?

A simple linear model assumes:

> the worker will continue moving in exactly the same direction and speed.

Human movement is not always linear.

A sequence model such as a GRU/LSTM can learn temporal motion patterns from recent observations.

For the hackathon, a physics-based baseline can also be implemented for comparison.

That allows the team to demonstrate:

```text
baseline extrapolation
vs
learned trajectory prediction
```

---

# 15. System output

The dashboard should display:

```text
Worker #4
Forklift #2

Current distance
Predicted future paths
Projected conflict point
Minimum predicted separation
Time to conflict
Risk score
```

When intervention is required:

```text
PREDICTED COLLISION IN 3.4 s

Recommended action:
Reduce vehicle speed by >=40%

Predicted risk before:
94%

Predicted risk after:
7%
```

---

# 16. Main demo

The strongest demonstration is deliberately counterintuitive.

### Scene

Worker and forklift are still far enough apart that a conventional proximity alarm would not trigger.

ForeSite displays:

```text
CURRENT DISTANCE
6.2 m

CURRENT PROXIMITY STATUS
SAFE
```

But the system predicts:

```text
FUTURE PATH CONFLICT
YES

TIME TO CONFLICT
3.4 s

PREDICTED MINIMUM DISTANCE
0.38 m

RISK
CRITICAL
```

Then:

```text
SIMULATING PREVENTIVE ACTIONS...
```

and finally:

```text
RECOMMENDED ACTION

REDUCE MACHINE SPEED BY >=40%
```

This directly demonstrates why the project is more than a proximity detector.

---

# 17. Possible simulation mode

Because real-world construction video can be unpredictable, ForeSite can also include a controlled simulation.

Example:

```text
+--------------------------------------+
|                                      |
|     👷 → → →                          |
|                                      |
|                       ↙              |
|                         🚜           |
|                                      |
+--------------------------------------+
```

The simulation allows:

- worker speed changes;
- machine speed changes;
- direction changes;
- guaranteed collision scenarios;
- deterministic intervention testing.

The same risk and counterfactual modules can be applied to the simulated trajectories.

This provides a robust backup demonstration.

---

# 18. Future extensions

Not required for the hackathon MVP:

- occlusion-aware "Ghost Worker" tracking;
- multi-camera fusion;
- ground-plane calibration;
- dynamic exclusion zones;
- PPE-risk integration;
- machine telematics;
- wearable alerts;
- digital twins;
- automatic braking integration;
- adaptive site-route planning.

---

# 19. What not to claim

Avoid:

> **No one has ever predicted construction collisions before.**

Avoid:

> **No system has ever automatically slowed machinery.**

Avoid:

> **This is the first construction AI safety system.**

Safer positioning:

> **ForeSite combines trajectory-based collision prediction with a prescriptive counterfactual layer that evaluates multiple preventive actions and identifies the least disruptive tested intervention capable of bringing predicted risk below the safety threshold.**

---

# 20. 30-second pitch

> **Construction safety systems usually detect danger after a worker has already entered an unsafe zone. ForeSite looks several seconds ahead. Using existing site video, it detects and tracks workers and heavy equipment and predicts their future trajectories. If those trajectories indicate an upcoming collision, ForeSite doesn't stop at a warning. It simulates possible preventive actions — such as slowing the machine or changing movement — and identifies the minimum intervention that brings the predicted risk back to a safe level. In short, ForeSite moves construction safety from detecting hazards to predicting and preventing accidents.**

---

# 21. 60-second pitch

> **On a construction site, a worker and a forklift can still be several metres apart and appear completely safe while already moving toward the same collision point. Most conventional safety systems react to current proximity. ForeSite AI instead uses computer vision to detect and track workers and machinery, then uses a trajectory-prediction model to forecast where both will move over the next few seconds. If the predicted trajectories create a dangerous interaction, our risk engine calculates the time to conflict and expected minimum separation. The key innovation comes next: ForeSite simulates several alternative futures — for example, slowing the forklift by 20%, 30%, or 40%, stopping it, or changing the worker's movement — and recommends the least disruptive tested action that reduces the predicted collision risk below a safe threshold. So rather than simply saying 'danger', ForeSite attempts to tell the site operator what action can prevent that danger from becoming an accident.**

---

# 22. Judge questions and suggested answers

## Q1. Isn't this just another YOLO construction-safety project?

**Answer:**

> YOLO is only our perception layer. Detection tells us what objects exist in the current frame. ForeSite's main contribution is downstream: persistent tracking builds movement histories, trajectory prediction forecasts where objects are going, the risk engine predicts future worker–machine conflicts, and the counterfactual layer evaluates alternative interventions to identify a minimum safe response.

---

## Q2. How is this different from a proximity sensor?

**Answer:**

> A proximity system reacts to current distance. Two objects can be close while moving away from each other, or far apart while moving rapidly toward the same point. ForeSite evaluates predicted future trajectories rather than current distance alone, so it can detect a dangerous interaction before the worker enters a fixed danger radius.

---

## Q3. Has collision prediction already been done?

**Answer:**

> Yes, trajectory forecasting and collision warning have already been explored, so we do not claim those independently as new. Our differentiator is the prescriptive layer: after detecting a future conflict, ForeSite tests multiple possible interventions and identifies the least disruptive tested action that makes the predicted situation safe.

---

## Q4. Why do you need a GRU/LSTM? Can't you just extrapolate velocity?

**Answer:**

> Constant-velocity extrapolation is a useful baseline, and we can keep it as a fallback. But workers and machines do not always move linearly. A sequence model uses recent temporal behavior to learn richer motion patterns. We can compare the learned predictor with the simple baseline to evaluate whether the model improves short-horizon forecasting.

---

## Q5. How do you calculate "40% speed reduction"?

**Answer:**

> Once a dangerous future is predicted, the intervention engine tests candidate speed reductions. For each candidate, it recalculates the future trajectory and risk. It chooses the smallest tested reduction that brings the risk below our predefined safety threshold. In the prototype, this is a search/optimization process rather than another black-box model.

---

## Q6. Why not always stop the machine?

**Answer:**

> A full stop may be safe, but automatically choosing the most disruptive response every time could make the system impractical and create alarm fatigue. ForeSite first enforces the safety constraint, then among safe options attempts to select the least disruptive intervention. In an emergency scenario, the minimum safe intervention may still be a full stop.

---

## Q7. Is the system actually controlling machinery?

**Answer:**

> The hackathon MVP is a decision-support system. It recommends an intervention to a safety officer or operator. Automatic machine control would require hardware integration, fail-safe engineering, and extensive safety certification, so that is positioned as future work rather than claimed in the prototype.

---

## Q8. What happens if the camera misses a worker?

**Answer:**

> ByteTrack helps bridge short detection gaps by maintaining object tracks. As a future enhancement, ForeSite can maintain a temporary estimated "ghost" position using the worker's recent trajectory, with decreasing confidence, rather than assuming that a temporarily occluded worker no longer represents a risk.

---

## Q9. What if the camera perspective makes distance inaccurate?

**Answer:**

> The first MVP operates primarily on normalized image coordinates and demonstrates the decision pipeline. A production system would include camera calibration and ground-plane projection or homography so image-space trajectories are translated into real-world distances and speeds.

This is an important limitation to state clearly.

---

## Q10. How will you validate it?

**Answer:**

> We can evaluate the trajectory model using displacement metrics such as ADE and FDE, and evaluate the safety layer using warning lead time, predicted minimum separation, risk before and after intervention, and whether the selected intervention prevents the simulated conflict. For the hackathon, controlled scenarios make these measurements repeatable.

---

## Q11. What happens if the prediction is wrong?

**Answer:**

> The system is intended as decision support, not as a replacement for standard construction safety systems. Predictions should be confidence-aware, and recommendations should only be issued when the estimated risk exceeds an appropriate threshold. Conservative fallback rules remain important in a safety-critical production system.

---

## Q12. Isn't a counterfactual result only hypothetical?

**Answer:**

> Yes — the intervention is an estimated future based on the system's motion and risk model. That is why we describe ForeSite as predictive decision support. The value is that the same model used to estimate the dangerous future is used consistently to compare candidate preventive futures.

---

## Q13. What is actually novel?

**Answer:**

> The novelty is not YOLO and not simply collision prediction. ForeSite connects real-time visual tracking and trajectory prediction to a prescriptive safety layer that generates and evaluates multiple preventive futures and selects the least disruptive safe intervention.

---

## Q14. Why construction rather than autonomous vehicles?

**Answer:**

> Construction sites have unusual combinations of pedestrians, heavy machinery, temporary pathways, poor visibility, changing layouts, and limited separation between people and equipment. Many sites also already have cameras, making visual safety intelligence a practical direction for retrofit-style deployment.

---

## Q15. Why use CCTV instead of wearable sensors?

**Answer:**

> CCTV allows the prototype to work with infrastructure many sites already have and does not require every worker to continuously carry or maintain a device. Wearables could complement ForeSite in future versions rather than being mutually exclusive.

---

## Q16. What happens at night or under poor visibility?

**Answer:**

> Vision performance will degrade under poor imaging conditions. Production deployment could use better low-light cameras, thermal imaging, multiple cameras, or fusion with other sensors. The hackathon prototype demonstrates the intelligence pipeline rather than claiming robustness to every operating condition.

---

## Q17. What about multiple workers and machines?

**Answer:**

> ByteTrack maintains multiple persistent tracks. The risk engine can evaluate every relevant worker–machine pair and rank them by predicted risk. For the demo we focus on one clear collision pair so the concept is easy to understand, but the architecture is designed to extend to multiple objects.

---

## Q18. What is the biggest limitation of the prototype?

**Answer:**

> The largest limitations are limited trajectory training data, image-space rather than fully calibrated real-world coordinates, and controlled intervention simulation rather than direct machinery control. These are deliberate hackathon scope choices; the core proof of concept is the end-to-end predictive and prescriptive workflow.

---

## Q19. Why is the project feasible in a hackathon?

**Answer:**

> We reuse mature components for perception and tracking rather than training everything from zero. YOLO handles detection, ByteTrack handles identity, a small GRU/LSTM handles short-horizon trajectory prediction, and the intervention engine uses lightweight simulation and search. That allows us to spend our innovation effort on integrating the full prevention workflow.

---

## Q20. How would this become a real product?

**Answer:**

A production path could be:

```text
Phase 1
Safety dashboard + alerts

Phase 2
Multi-camera calibrated site tracking

Phase 3
Integration with operator tablets / cabin alerts

Phase 4
Machine telematics + controlled slowdown

Phase 5
Certified automatic safety intervention
```

---

# 23. Questions judges may ask your CV lead specifically

## Why YOLO?

> YOLO provides a strong speed/accuracy balance for real-time video and is straightforward to integrate with tracking.

## Why ByteTrack?

> It provides lightweight multi-object association across frames and integrates naturally with YOLO detections, which is important because the trajectory model requires stable object identities.

## Why bottom-center coordinates?

> For a pedestrian bounding box, the bottom-center is a better image-space approximation of where the worker contacts the ground than the geometric center. That gives a more meaningful trajectory for movement analysis.

## How do you avoid resolution dependence?

> We normalize coordinates by frame width and height before passing trajectories to the prediction model.

## What happens during temporary occlusion?

> Track history is retained for a short period rather than deleted immediately, and ByteTrack can recover short gaps. Longer occlusion handling is future work.

---

# 24. Key presentation message

The audience should remember this progression:

```text
Traditional CCTV
RECORDS

Safety detection AI
DETECTS

Predictive safety AI
FORECASTS

ForeSite
PREDICTS + PRESCRIBES
```

---

# 25. Final closing line

> **ForeSite doesn't just ask whether a construction site is dangerous now. It asks what is likely to become dangerous next — and what action can prevent it.**
