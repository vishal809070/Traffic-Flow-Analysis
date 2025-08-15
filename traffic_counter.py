import os
import cv2
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from shapely.geometry import Point, Polygon
from ultralytics import YOLO



def download_youtube(url: str, out_dir: str) -> str:
    """Download a YouTube video using yt-dlp and return path to mp4 file."""
    try:
        import yt_dlp 
    except Exception as e:
        raise RuntimeError(
            "yt-dlp is required to download YouTube videos. Install it with: pip install yt-dlp"
        ) from e

    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "source.%(ext)s")
    ydl_opts = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": out_template,
        "quiet": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    
    for name in os.listdir(out_dir):
        if name.lower().endswith(".mp4"):
            return os.path.join(out_dir, name)
    
    files = [os.path.join(out_dir, f) for f in os.listdir(out_dir)]
    if not files:
        raise FileNotFoundError("Download finished but no file found.")
    return files[0]



def denormalize_point(pt: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
    return int(pt[0] * w), int(pt[1] * h)

def load_and_denormalize_lanes(lanes_file: str, W: int, H: int):
    """Load normalized lanes and count line; return pixel-space Polygons and line points."""
    with open(lanes_file, "r") as f:
        data = json.load(f)

    lanes = []
    for lane in data["lanes"]:
        pts_px = [denormalize_point(p, W, H) for p in lane["polygon"]]
        lanes.append({
            "id": lane["id"],
            "polygon": Polygon(pts_px),   
            "pts_px": np.array(pts_px, dtype=np.int32)  
        })

    count_line_px = (
        denormalize_point(data["count_line"][0], W, H),
        denormalize_point(data["count_line"][1], W, H),
    )
    return lanes, count_line_px



def analyze(
    source_url: str,
    lanes_file: str,
    show: bool = False,
    save_video: bool = True,
    classes_to_detect: List[int] = [2, 3, 5, 7],  
    conf_thres: float = 0.3,
    imgsz: int = 640,
    tracker_cfg: str = "botsort.yaml"  
):

    os.makedirs("outputs", exist_ok=True)

    print("[info] Downloading YouTube video...")
    src_path = download_youtube(source_url, "outputs")
    print(f"[ok] Downloaded: {src_path}")

    
    print("[info] Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] Video size: {W}x{H} @ {fps:.2f} FPS")

    
    lanes, count_line = load_and_denormalize_lanes(lanes_file, W, H)
    y_line = count_line[0][1]  
    y_tol = max(5, H // 100)  

    
    out_writer = None
    out_path = None
    if save_video:
        out_path = os.path.join("outputs", "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    
    counted_ids = set()  
    vehicle_counts = {lane["id"]: 0 for lane in lanes}
    csv_rows: List[Dict[str, Any]] = []

    frame_idx = 0
    t_prev = time.time()
    fps_smooth = None

    print("[info] Processing... (press 'q' to quit if --show)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        
        results = model.track(
            frame,
            persist=True,
            classes=classes_to_detect,
            conf=conf_thres,
            imgsz=imgsz,
            tracker=tracker_cfg,
            verbose=False
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                pt = Point(cx, cy)

                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(track_id)}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                
                for lane in lanes:
                    if lane["polygon"].contains(pt):
                        if abs(cy - y_line) <= y_tol and int(track_id) not in counted_ids:
                            counted_ids.add(int(track_id))
                            vehicle_counts[lane["id"]] += 1
                            csv_rows.append({
                                "vehicle_id": int(track_id),
                                "lane": lane["id"],
                                "frame": frame_idx,
                                "timestamp": round(frame_idx / fps, 3)
                            })
                        break  

        
        for lane in lanes:
            cv2.polylines(frame, [lane["pts_px"]], isClosed=True, color=(255, 0, 0), thickness=2)
            x0, y0 = lane["pts_px"][0]
            cv2.putText(frame, f"Lane {lane['id']}: {vehicle_counts[lane['id']]}",
                        (int(x0), max(0, int(y0) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        
        cv2.line(frame, count_line[0], count_line[1], (0, 0, 255), 2)

       
        t_now = time.time()
        inst = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_smooth = inst if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if show:
            cv2.imshow("Traffic Flow (YOLOv8 + SORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_writer is not None:
            out_writer.write(frame)

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

    
    csv_path = os.path.join("outputs", "vehicle_counts.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    
    print("\n--- Final Summary ---")
    for lane_id, c in vehicle_counts.items():
        print(f"Lane {lane_id}: {c} vehicles")
    print(f"\nCSV saved to: {csv_path}")
    if out_path:
        print(f"Annotated video saved to: {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis (YOLOv8 + SORT-family)")
    parser.add_argument("--url",
                        default="https://www.youtube.com/watch?v=MNn9qKG2UFI",
                        help="YouTube video URL to download and process")
    parser.add_argument("--lanes", default="lanes.json", help="Path to lanes.json (normalized coords)")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--save_video", action="store_true", help="Save annotated.mp4")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size (e.g., 480/640)")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config (botsort.yaml or bytetrack.yaml)")
    args = parser.parse_args()

    analyze(
        source_url=args.url,
        lanes_file=args.lanes,
        show=args.show,
        save_video=args.save_video,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        tracker_cfg=args.tracker,
    )
