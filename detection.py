from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on video
results = model("property_walkthrough.mp4", save=True, project="runs", name="detect_test")

print("✅ Detection complete! Check the 'runs/detect_test' folder for the output video.")

# Print detected objects
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"{label}: {conf:.2f}")
