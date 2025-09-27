# detection.py
import os
import sys
import cv2
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except Exception:
    DEEPSORT_AVAILABLE = False

# ---------------- Config ----------------
MODEL_WEIGHTS = "yolov8n.pt"
DEFAULT_INPUT = "property_walkthrough3.mp4"
CROP_DIR = "crops"
os.makedirs(CROP_DIR, exist_ok=True)

FURNITURE_CLASSES = {"chair", "couch", "sofa", "bed", "dining table", "tv", "bench"}
CONF_THRESHOLD = 0.3
# ----------------------------------------

if len(sys.argv) > 1:
    INPUT_PATH = sys.argv[1]
else:
    INPUT_PATH = DEFAULT_INPUT

print(f"Input: {INPUT_PATH}")
model = YOLO(MODEL_WEIGHTS)

def label_name(cls_id):
    return str(model.names[int(cls_id)]).replace(" ", "_")

# ---------- IMAGE PROCESSING ----------
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: could not read image:", img_path)
        return

    result = model(img)[0]
    counts = {}
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        cls_id = int(box.cls[0])
        label = label_name(cls_id)
        if label not in FURNITURE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        counts[label] = counts.get(label, 0) + 1
        out_fn = os.path.join(CROP_DIR, f"{label}_{counts[label]}.jpg")
        cv2.imwrite(out_fn, crop)
        print("Saved crop:", out_fn, f"(conf={conf:.2f})")

    # save annotated image
    annotated = result.plot()
    annotated_name = "annotated_" + os.path.basename(img_path)
    cv2.imwrite(annotated_name, annotated)
    print("Saved annotated image:", annotated_name)

# ---------- VIDEO PROCESSING ----------
def process_video(video_path):
    if not DEEPSORT_AVAILABLE:
        print("ERROR: deep-sort-realtime not installed. Run: pip install deep-sort-realtime")
        return

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w, out_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    annotated_out = cv2.VideoWriter("annotated_" + os.path.basename(video_path), fourcc, fps, (out_w, out_h))

    tracker = DeepSort(max_age=30)
    saved_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame)[0]
        detections = []
        det_list = []

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            label = label_name(cls_id)
            if label not in FURNITURE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 4 or bh <= 4:
                continue

            detections.append(([x1, y1, bw, bh], conf, cls_id))
            cx, cy = x1 + bw / 2, y1 + bh / 2
            det_list.append({'bbox': [x1, y1, bw, bh], 'conf': conf, 'cls': cls_id, 'center': (cx, cy)})

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            tcx, tcy = (l + r) / 2, (t + b) / 2

            assigned_label = None
            for d in det_list:
                cx, cy = d['center']
                if l <= cx <= r and t <= cy <= b:
                    assigned_label = label_name(d['cls'])
                    break
            if assigned_label is None:
                assigned_label = "object"

            if track_id not in saved_ids:
                l, t, r, b = max(0, l), max(0, t), min(out_w - 1, r), min(out_h - 1, b)
                if r <= l or b <= t:
                    continue
                crop = frame[t:b, l:r]
                out_fn = os.path.join(CROP_DIR, f"{assigned_label}_{track_id}.jpg")
                cv2.imwrite(out_fn, crop)
                saved_ids.add(track_id)
                print("Saved unique crop:", out_fn)

            # draw annotation on frame
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"{assigned_label}_{track_id}", (int(l), int(t) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        annotated_out.write(frame)

    cap.release()
    annotated_out.release()
    print("Video processing complete. Crops in", CROP_DIR, "Annotated video saved as:",
          "annotated_" + os.path.basename(video_path))

# ---------- MAIN ----------
ext = os.path.splitext(INPUT_PATH)[1].lower()
if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
    process_image(INPUT_PATH)
else:
    process_video(INPUT_PATH)

print("Done.")
