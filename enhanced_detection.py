"""
Enhanced Detection Script
Based on your original detection.py but with improved condition classification
"""

import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np

# ---------------- Config ----------------
MODEL_WEIGHTS = "improved_furniture_runs/furniture_condition_v2/weights/best.pt"  # Use improved model
DEFAULT_INPUT = "property_walkthrough3.mp4"
CROP_DIR = "crops"
os.makedirs(CROP_DIR, exist_ok=True)

# Output directories for annotations
ANNOTATED_IMAGES_DIR = "annotated_images"
ANNOTATED_VIDEOS_DIR = "annotated_videos"
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_VIDEOS_DIR, exist_ok=True)

# Furniture classes with condition mapping
FURNITURE_CLASSES = {
    0: "chair_new",
    1: "chair_broken", 
    2: "chair_wornout",
    3: "sofa_new",
    4: "sofa_broken",
    5: "sofa_wornout",
    6: "table_new",
    7: "table_broken",
    8: "table_wornout"
}

CONF_THRESHOLD = 0.3

# Condition colors for visualization
CONDITION_COLORS = {
    "new": (0, 255, 0),      # Green
    "broken": (0, 0, 255),   # Red  
    "wornout": (0, 165, 255) # Orange
}

def get_condition_from_class(class_id):
    """Get condition from class ID"""
    class_name = FURNITURE_CLASSES.get(class_id, "unknown")
    if "new" in class_name:
        return "new"
    elif "broken" in class_name:
        return "broken"
    elif "wornout" in class_name:
        return "wornout"
    else:
        return "unknown"

def get_furniture_type(class_id):
    """Get furniture type from class ID"""
    class_name = FURNITURE_CLASSES.get(class_id, "unknown")
    if "chair" in class_name:
        return "chair"
    elif "sofa" in class_name:
        return "sofa"
    elif "table" in class_name:
        return "table"
    else:
        return "unknown"

def process_image(img_path):
    """Process a single image with improved condition detection"""
    print(f"Processing image: {img_path}")
    
    # Load model
    if os.path.exists(MODEL_WEIGHTS):
        model = YOLO(MODEL_WEIGHTS)
        print("Using improved condition detection model")
    else:
        model = YOLO("yolov8n.pt")
        print("Using basic YOLO model (no condition classification)")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    
    # Run detection
    results = model(img, conf=CONF_THRESHOLD)
    
    # Process results
    detections = []
    annotated_img = img.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get furniture type and condition
                furniture_type = get_furniture_type(class_id)
                condition = get_condition_from_class(class_id)
                
                # Store detection
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'furniture_type': furniture_type,
                    'condition': condition,
                    'class_id': class_id
                }
                detections.append(detection)
                
                # Draw bounding box with condition-specific color
                color = CONDITION_COLORS.get(condition, (128, 128, 128))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{furniture_type.upper()}: {condition.upper()} ({confidence:.2f})"
                cv2.putText(annotated_img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Crop furniture for detailed analysis
                furniture_crop = img[y1:y2, x1:x2]
                if furniture_crop.size > 0:
                    crop_filename = f"{furniture_type}_{condition}_{i}_{os.path.basename(img_path)}"
                    crop_path = os.path.join(CROP_DIR, crop_filename)
                    cv2.imwrite(crop_path, furniture_crop)
                    print(f"  Cropped {furniture_type} ({condition}) to: {crop_path}")
    
    # Save annotated image
    annotated_filename = f"annotated_{os.path.basename(img_path)}"
    annotated_path = os.path.join(ANNOTATED_IMAGES_DIR, annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)
    
    # Print results
    print(f"Found {len(detections)} furniture items:")
    for det in detections:
        print(f"  - {det['furniture_type']}: {det['condition']} (confidence: {det['confidence']:.2f})")
    
    print(f"Annotated image saved to: {annotated_path}")
    return detections

def process_video(video_path):
    """Process video with furniture condition detection"""
    print(f"Processing video: {video_path}")
    
    # Load model
    if os.path.exists(MODEL_WEIGHTS):
        model = YOLO(MODEL_WEIGHTS)
        print("Using improved condition detection model")
    else:
        model = YOLO("yolov8n.pt")
        print("Using basic YOLO model (no condition classification)")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(ANNOTATED_VIDEOS_DIR, f"annotated_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection on every 5th frame (for performance)
        if frame_count % 5 == 0:
            results = model(frame, conf=CONF_THRESHOLD)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        furniture_type = get_furniture_type(class_id)
                        condition = get_condition_from_class(class_id)
                        
                        # Draw bounding box
                        color = CONDITION_COLORS.get(condition, (128, 128, 128))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{furniture_type.upper()}: {condition.upper()}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        total_detections += 1
        
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"Video processing completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Annotated video saved to: {output_path}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = DEFAULT_INPUT
    
    print("Enhanced Furniture Condition Detection")
    print("=" * 50)
    print(f"Input: {input_path}")
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(input_path)
        else:
            process_image(input_path)
    else:
        print(f"File not found: {input_path}")

if __name__ == "__main__":
    main()
