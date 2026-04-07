# 🎯 Multi-Object Tracking — Sports Video
> Detect and persistently track all players in a 60-second football clip using YOLOv8m + ByteTrack.

---

## 🚀 Live Demo

| | |
|---|---|
| **Live App** | [sportstracking.streamlit.app](https://sportstracking.streamlit.app/) |
| **Demo Video** |[Demo Video]([(https://youtu.be/Ji423oQ3SnE))|

---

## 📌 Overview

- **Goal:** Assign a persistent ID to every active player across the full video
- **Model:** YOLOv8m (person class only) + ByteTrack for association
- **Stack:** Python · Ultralytics · Supervision · OpenCV · Streamlit

---

## ❗ Problem — ID Switching

- Baseline (default ByteTrack settings) produced **79 unique IDs** for ~22 actual players
- Root causes: re-entry spawning new IDs, single-frame ghost detections, low-confidence occlusion mismatches
- Target: reduce ID count to reflect the true number of players on the pitch

---

## 🧠 Approach

| Component | Choice | Why |
|---|---|---|
| Detector | YOLOv8m | Best speed/accuracy tradeoff; nano missed occluded players |
| Tracker | ByteTrack | Handles occlusion via low-confidence track buffer |
| Filter | Area threshold (≥ 1 000 px²) | Removes partial-body edge detections |
| Stationary filter | Centroid drift < 20 px | Suppresses background viewers without touching tracker state |

---

## 🔧 Pipeline

```
Input video
  └─▶ Frame decode (OpenCV)
        └─▶ YOLOv8m detection  [conf ≥ 0.30, class = person]
              └─▶ Area filter  [drop boxes < 1 000 px²]
                    └─▶ ByteTrack association
                          └─▶ Stationary viewer suppression
                                └─▶ Annotated frame → output video
```

---

##  Improvements Made

- **`track_activation_threshold` 0.50 → 0.65** — stops re-entering players from spawning a fresh ID
- **`minimum_consecutive_frames` 1 → 3** — eliminates single-frame ghost detections
- **`lost_track_buffer` kept at 60 frames** — bridges 2-second occlusions without stale-track interference
- **`match_thresh` raised to 0.75** — tighter IoU matching reduces cross-player ID swaps in crowds
- **Area filter (≥ 1 000 px²)** — drops partial bodies at frame edges before they enter the tracker
- **Stationary filter (post-process)** — greys out background viewers, keeping tracker state clean

---

## 📊 Results

| Metric | Baseline | Optimised |
|---|---|---|
| Unique IDs assigned | 79 | **~47** |
| ID reduction | — | **~40%** |
| Frames processed | 1 801 | 1 801 |
| Video duration | 60 s @ 30 fps | 60 s @ 30 fps |
| Processing speed | ~5.4 FPS (CPU) | ~5.4 FPS (CPU) |
| Viewer IDs suppressed | — | visible as grey boxes |

---

## ▶️ Run Instructions

```bash
# 1. Clone & install
git clone https://github.com/Yuvrajtakk/multi-object-tracking-assignment.git
cd multi-object-tracking-assignment
pip install -r requirements.txt

# 2. Add your video
cp your_video.mp4 input/

# 3a. Run the Streamlit app
streamlit run app.py

# 3b. Or run the notebook
jupyter notebook mot_pipeline.ipynb
```

---

## 📁 Input / Output

```
multi-object-tracking-assignment/
├── input/
│   └── 15sec_clip.mp4          ← source video (place yours here)
├── output/
│   └── 15sec_clip_tracked.mp4  ← annotated output
├── mot_pipeline.ipynb           ← full experiment notebook
├── app.py                       ← Streamlit deployment
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- **Re-entry > 2 s** — tracks expire after 60 frames; returning players get a new ID
- **Dense crowds** — IoU matching degrades when 3+ players overlap simultaneously
- **CPU-only** — ~5 FPS; real-time requires GPU (CUDA)
- **No appearance ReID** — purely motion-based; identical-jersey players can swap IDs

---

## 🔮 Future Work

- [ ] Add appearance embeddings (BoT-SORT / StrongSORT) to handle re-entry correctly
- [ ] Fine-tune YOLOv8 on sports-specific data to improve occlusion detection
- [ ] Team classification via jersey-colour clustering
- [ ] Real-time GPU inference with TensorRT export

---

## ☑️ Submission Checklist

- [x] Input video placed in `input/`
- [x] Tracked output video in `output/`
- [x] Notebook `mot_pipeline.ipynb` — fully executed with outputs
- [x] Streamlit app live at [sportstracking.streamlit.app](https://sportstracking.streamlit.app/)
- [x] Unique ID count reduced from **79 → ~47** (~40% improvement)
- [x] Stationary viewer suppression implemented
- [x] All parameters documented and tunable via sidebar
