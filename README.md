# Multi-Object Tracking — Streamlit App

Detects and tracks all players in a sports video using **YOLOv8m + ByteTrack**.

## Features
- Upload any MP4 / AVI / MOV / MKV video
- Sidebar controls for all detection & tracking parameters
- Mid-video detection preview before running the full pipeline
- Stationary-viewer suppression (greyed-out background persons)
- Live progress bar + per-frame metrics while processing
- Sample frame preview + one-click download of the tracked video

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io → **New app**.
3. Point it at `app.py` in your repo.
4. Done — Streamlit Cloud will install `requirements.txt` automatically.

> **Note:** YOLOv8m runs on CPU in the cloud (~5 FPS).  
> For GPU-accelerated processing, deploy on a machine with CUDA and replace  
> `opencv-python-headless` with `opencv-python` in `requirements.txt`.

## File structure

```
mot_app/
├── app.py            # Streamlit application
├── requirements.txt  # Python dependencies
└── README.md
```
