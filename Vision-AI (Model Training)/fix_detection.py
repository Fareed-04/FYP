"""
Fixed Detection Script
Based on your original detection.py but with working condition classification
"""

import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np

# ---------------- Config ----------------
MODEL_WEIGHTS = "yolov8n.pt"  # Use pre-trained model for now
DEFAULT_INPUT = "property_walkthrough3.mp4"
CROP_DIR = "crops"
os.makedirs(CROP_DIR, exist_ok=True)

# Output directories for annotations
ANNOTATED_IMAGES_DIR = "annotated_images"
ANNOTATED_VIDEOS_DIR = "annotated_videos"
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_VIDEOS_DIR, exist_ok=True)

FURNITURE_CLASSES = {"chair", "couch", "sofa", "bed", "dining table", "tv", "bench"}
CONF_THRESHOLD = 0.3

def classify_condition(image, bbox):
    """Classify furniture condition based on visual analysis"""
    x1, y1, x2, y2 = bbox
    furniture_crop = image[y1:y2, x1:x2]
    
    if furniture_crop.size == 0:
        return "unknown", 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(furniture_crop, cv2.COLOR_BGR2GRAY)
    
    # Calculate features
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Edge detection for damage assessment
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Color analysis
    hsv = cv2.cvtColor(furniture_crop, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])
    
    # Classification logic (improved)
    if edge_density > 0.2 or mean_saturation < 30:
        return "broken", 0.85
    elif mean_brightness < 80 or mean_value < 100 or mean_saturation < 60:
        return "wornout", 0.75
    else:
        return "new", 0.90

def process_image(img_path):
    """Process a single image with furniture condition detection"""
    print(f"Processing image: {img_path}")
    
    # Load model
    model = YOLO(MODEL_WEIGHTS)
    
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
                class_name = model.names[class_id]
                
                # Only process furniture
                if class_name.lower() in FURNITURE_CLASSES:
                    # Classify condition
                    condition, condition_conf = classify_condition(img, [x1, y1, x2, y2])
                    
                    # Store detection
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_name': class_name,
                        'condition': condition,
                        'condition_confidence': condition_conf
                    }
                    detections.append(detection)
                    
                    # Choose color based on condition
                    if condition == "new":
                        color = (0, 255, 0)      # Green
                    elif condition == "broken":
                        color = (0, 0, 255)      # Red
                    elif condition == "wornout":
                        color = (0, 165, 255)    # Orange
                    else:
                        color = (128, 128, 128)  # Gray
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name.upper()}: {condition.upper()} ({confidence:.2f})"
                    cv2.putText(annotated_img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add condition confidence
                    conf_label = f"Condition: {condition_conf:.2f}"
                    cv2.putText(annotated_img, conf_label, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Crop furniture for detailed analysis
                    furniture_crop = img[y1:y2, x1:x2]
                    if furniture_crop.size > 0:
                        crop_filename = f"{class_name}_{condition}_{i}_{os.path.basename(img_path)}"
                        crop_path = os.path.join(CROP_DIR, crop_filename)
                        cv2.imwrite(crop_path, furniture_crop)
                        print(f"  Cropped {class_name} ({condition}) to: {crop_path}")
    
    # Save annotated image
    annotated_filename = f"annotated_{os.path.basename(img_path)}"
    annotated_path = os.path.join(ANNOTATED_IMAGES_DIR, annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)
    
    # Print results
    print(f"Found {len(detections)} furniture items:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['condition']} "
              f"(detection: {det['confidence']:.2f}, condition: {det['condition_confidence']:.2f})")
    
    print(f"Annotated image saved to: {annotated_path}")
    return detections

def process_video(video_path):
    """Process video with furniture condition detection"""
    print(f"Processing video: {video_path}")
    
    # Load model
    model = YOLO(MODEL_WEIGHTS)
    
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
        
        # Run detection on every 10th frame (for performance)
        if frame_count % 10 == 0:
            results = model(frame, conf=CONF_THRESHOLD)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        if class_name.lower() in FURNITURE_CLASSES:
                            # Classify condition
                            condition, condition_conf = classify_condition(frame, [x1, y1, x2, y2])
                            
                            # Choose color
                            if condition == "new":
                                color = (0, 255, 0)
                            elif condition == "broken":
                                color = (0, 0, 255)
                            elif condition == "wornout":
                                color = (0, 165, 255)
                            else:
                                color = (128, 128, 128)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label
                            label = f"{class_name.upper()}: {condition.upper()}"
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
    
    print("Fixed Furniture Condition Detection")
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
