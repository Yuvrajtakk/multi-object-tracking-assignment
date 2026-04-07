# Multi-Object Tracking — Football Player Detection & Persistent ID Assignment

**Video source:** [Football Match Clip — Roboflow Universe](https://universe.roboflow.com/)  
**Stack:** YOLOv8m · ByteTrack · supervision · OpenCV · Python

---

## Overview

A CPU-based tracking pipeline that detects every player in a 60-second football match clip and assigns each one a persistent unique ID — stable across the full video even through occlusions, fast movement, and player re-entries.

The focus of this project was not just detection, but **ID consistency**. The baseline approach gave 79 unique IDs for a scene with ~22 players. After systematic parameter tuning and analysis, the final pipeline reduces that to ~47 IDs — a 40% improvement.

---

## Problem

Detection is solved. Tracking is hard.

A naive approach detects players correctly on every frame but treats each frame independently. In a 60-second clip with 22+ players constantly overlapping and crossing each other, this means a new ID every time a player is briefly occluded. The real challenge is making sure Player 7 is still ID 7 five seconds later after disappearing behind a group.

An additional challenge: the camera captures viewers and staff in the background who are valid person detections. Without filtering, they inflate the ID count with stationary background noise.

---

## Approach

**Detection — YOLOv8m**  
The medium model was the right size here. Nano misses players at the far end of the pitch (small bounding boxes, partial occlusion). Extra-large is too slow on CPU. Medium runs at ~5 FPS and catches enough to feed a reliable tracker.

**Tracking — ByteTrack (via supervision)**  
ByteTrack runs two-stage IoU matching: high-confidence detections are matched first, then low-confidence detections are used to update tracks of partially occluded players. No ReID network needed — pure position and overlap. This is the right call when compute is limited and occlusions are short.

The combination is deliberately lightweight. The goal was a pipeline that works well, not one that's hard to run.

---

## Pipeline

```
input/ folder
    ↓
Frame read (OpenCV)
    ↓
Person detection (YOLOv8m · class=0 · confidence filter)
    ↓
Area filter (drop boxes under MIN_BOX_AREA px²)
    ↓
ByteTrack (IoU matching · Kalman prediction · lost-track buffer)
    ↓
ID annotation (colour-coded boxes · persistent ID labels)
    ↓
output/ folder
```

---

## Key Improvements (79 → 47 IDs)

Five full runs were completed, each analysing the per-100-frame ID accumulation log to find what was going wrong.

| Run | Key Change | IDs |
|-----|-----------|-----|
| Baseline | `conf=0.25`, `match=0.8`, `buffer=30` | 79 |
| Run 2 | Lowered `match=0.7`, raised `buffer=90` | 86 ↑ worse |
| Run 3 | Tighter area filter, `min_hits=3` | 84 |
| Run 4 | Added NMS, dropped `match=0.6` | 104 ↑ much worse |
| **Run 5** | **Raised `track_activation_threshold=0.65`, `match=0.75`** | **47 ✓** |

**The single biggest fix** was `track_activation_threshold`. It had sat at the default `0.5` across every run. Raising it to `0.65` means re-entering players at moderate confidence (0.5–0.6) can update their existing track instead of immediately spawning a fresh ID. Combined with restoring `match_thresh` toward `0.75` (the data clearly showed `0.8 > 0.7 > 0.6` for this video), this produced the largest single-run drop.

Lowering `match_thresh` consistently made things worse. In crowded scenes a looser IoU requirement causes wrong player-to-track pairings — the right player fails to match and gets a new ID. This is the opposite of the intuition.

**Final parameters:**

| Parameter | Value | Role |
|-----------|-------|------|
| `CONF_THRESH` | 0.30 | YOLO detection floor |
| `MIN_BOX_AREA` | 1000 px² | Drops tiny partial detections |
| `TRACK_THRESH` | 0.50 | Min confidence to start a new track |
| `MATCH_THRESH` | 0.75 | IoU required to link detection to existing track |
| `BUFFER_SIZE` | 60 frames | Lost track memory (2 seconds @ 30fps) |
| `MIN_HITS` | 3 frames | Consecutive frames before ID is assigned |

---

## Results

| Metric | Value |
|--------|-------|
| Video | 1280×720 · 30fps · 60s · 1801 frames |
| Processing speed | ~5.4 FPS (CPU) |
| Unique IDs — baseline | 79 |
| Unique IDs — final | ~47 |
| Reduction | ~40% |

~47 IDs for a 22-player scene is a reasonable result for IoU-only tracking on CPU. The remaining excess IDs are almost entirely players who leave frame for more than 2 seconds — the buffer window — and re-enter as new persons.

---

## How to Run

**1. Install dependencies**
```bash
pip install ultralytics supervision opencv-python numpy matplotlib
```

**2. Add your video**
```
Place any .mp4 / .avi / .mov / .mkv file into the input/ folder
```

**3. Run the notebook**
```bash
jupyter notebook mot_pipeline_final.ipynb
```
Run all cells top to bottom. Output is saved automatically to `output/`.

No path editing required — the pipeline detects any video in `input/` automatically.

---

## Input / Output

```
project/
├── input/          ← drop your video here (any common format)
├── output/         ← annotated video saved here automatically
├── mot_pipeline_final.ipynb
└── README.md
```

Input video is detected automatically. Output filename is `{input_name}_tracked.mp4`.

---

## Limitations

**Re-entry after long absence** — if a player leaves frame for more than 2 seconds (60 frames), their track expires. On return they get a new ID. This is the main remaining source of excess IDs and requires appearance-based ReID to fix properly.

**Crowded scenes** — when 3+ players overlap for several frames, IoU matching can swap IDs. `match_thresh=0.75` keeps this under control for most cases.

**Camera motion** — fast pans cause IoU matching to degrade because Kalman-predicted positions diverge from actual positions. BoT-SORT with Global Motion Compensation was tested but produced 297 IDs on this clip due to untuned GMC parameters.

**CPU speed** — ~5 FPS means roughly 5–6 minutes to process a 60-second video.

---

## Future Improvements

- **Appearance ReID (OSNet / FastReID)** — would allow re-matching players by appearance after long absences, directly solving the main remaining ID inflation source
- **GPU inference** — switching `device="cuda"` in the YOLO call brings speed to near real-time (~25–40 FPS)
- **Trajectory visualisation** — drawing movement trails per ID over a rolling window
- **Player count over time** — plotting active IDs per frame to show substitutions or tactical shifts
- **Team clustering** — jersey colour clustering to separate the two teams automatically

---

## Technical Report

Full write-up covering model choices, parameter experimentation, failure cases, and next steps is included in the notebook's final markdown cell.

---

## Repository

[github.com/Yuvrajtakk/multi-object-tracking-assignment](https://github.com/Yuvrajtakk/multi-object-tracking-assignment)
