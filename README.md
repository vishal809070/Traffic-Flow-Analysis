# Traffic Flow Analysis — 3-Lane Vehicle Counter (YOLOv8 + BoT-SORT)

This project counts vehicles per lane from a traffic video using **YOLOv8** detections and **BoT-SORT** MOT (via Ultralytics). It tracks vehicles across frames and counts each vehicle **once** as it crosses a virtual counting line, assigning the count to the lane in which the vehicle crosses.

Video used: https://www.youtube.com/watch?v=MNn9qKG2UFI

Demo Video Link:- https://drive.google.com/drive/folders/1bBDpjIlK_XolNfzUO5kVFbCkrdGAc6uT

## ✨ Features
- Download video from YouTube automatically (via `yt-dlp`).
- Detect with pre-trained COCO model (`yolov8n.pt`).
- Track with **ByteTrack** to avoid double counting.
- Define **three lane polygons** + **one counting line** in `lanes.json`.
- Real-time overlays: lanes, count line, live totals per lane.
- CSV export: **vehicle_id, lane, frame, timestamp**.
- Optional: save annotated output video (`outputs/annotated.mp4`).

## 🧱 Tech Choices
- **Ultralytics YOLOv8** for accurate and fast detection with easy tracking API.
- **ByteTrack** (Ultralytics tracker backend) for stable multi-object tracking.
- **OpenCV** to draw overlays and do geometry (point-in-polygon).
- **yt-dlp** to fetch the YouTube video programmatically.
- **Pandas** to export clean CSVs.

## 📦 Setup

> Python 3.10+ recommended.

```bash
# 1) Create venv (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
# If PyTorch didn't install automatically for your platform, follow:
# https://pytorch.org/get-started/locally/
```

## 🚦 Configure Lanes and Count Line

Edit **lanes.json**. Coordinates are **normalized** (0..1), so they scale to any video resolution.

- `lanes`: three polygons, each list is `[x, y]` point (clockwise or counterclockwise).
- `count_line`: two points `[x, y]`; a vehicle is counted when its **track** crosses this line.

> Tip: Run once with `--show` to preview overlays and adjust polygons until they match the three real lanes in the video. The defaults provided are a starting point only.

## ▶️ Run

```bash
# Basic run (will download the YouTube video on first run)
python traffic_counter.py --show --save_video

# Faster test (process ~1000 frames, don’t save video)
python traffic_counter.py --max_frames 1000

# Force CPU (if no CUDA GPU available)
python traffic_counter.py --device cpu --save_video

# Use a bigger model for accuracy (slower)
python traffic_counter.py --model yolov8s.pt --save_video
```

Outputs are written to **outputs/**:
- `annotated.mp4` – overlay video (if `--save_video` used)
- `counts.csv` – rows: vehicle_id, lane, frame, timestamp

## 📊 CSV Schema

| column      | type   | description                                  |
|-------------|--------|----------------------------------------------|
| vehicle_id  | int    | Persistent tracker ID from ByteTrack         |
| lane        | int    | 1, 2, or 3                                   |
| frame       | int    | Video frame index at the moment of counting  |
| timestamp   | str    | Wall-time since video start (HH:MM:SS)       |

## 🎥 Demo Video

- The script can save `outputs/annotated.mp4` automatically.
- Trim to 60–120 seconds using your editor or FFmpeg:
  ```bash
  ffmpeg -i outputs/annotated.mp4 -ss 00:00:00 -t 00:01:30 -c copy demo.mp4
  ```
- Upload `demo.mp4` to Google Drive and share a view link.

## 🧪 Accuracy & Performance Tips

- Keep the `count_line` where vehicles are well-separated and moving in one direction.
- Use `yolov8s.pt` or `yolov8m.pt` if accuracy is more important than FPS.
- On laptops without GPU, reduce load:
  - Use `--model yolov8n.pt`
  - Use a lower input resolution or set `--max_frames` while testing.
- Track stability: ByteTrack is default; you can switch to BoT-SORT by setting `tracker="botsort.yaml"` in the script for more robust ID persistence (slower).

## 🧰 Project Structure

```
traffic-flow-counter/
├── lanes.json          # lane polygons + counting line (normalized)
├── requirements.txt
├── traffic_counter.py  # main script
└── outputs/            # created at runtime
```

## 📝 Submission Checklist

- ✅ GitHub repo with this code and README.
- ✅ Commit `lanes.json` tuned for your video.
- ✅ Include `outputs/demo.mp4` (1–2 min) in repo or Google Drive link.
- ✅ Provide the repo link + demo link + short technical summary:
  - Detector: YOLOv8n (COCO). Classes: car(2), motorcycle(3), bus(5), truck(7).
  - Tracker: ByteTrack via Ultralytics. Count when crossing a line; lane via point-in-polygon.
  - CSV + visual overlays + final per-lane totals.
