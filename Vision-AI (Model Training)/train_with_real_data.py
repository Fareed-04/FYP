"""
Train Furniture Condition Detection with Real Data
This script will train a model using the images you add to furniture_dataset/
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
import random

def create_yolo_dataset():
    """Create YOLO dataset from furniture_dataset folder"""
    print("Creating YOLO dataset from real images...")
    
    # Create YOLO dataset structure
    dataset_dir = Path("real_furniture_dataset")
    (dataset_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # Define classes and conditions
    furniture_types = ["chair", "sofa", "table"]
    conditions = ["broken", "wornout"]
    
    # Create class mapping
    class_mapping = {}
    class_id = 0
    for furniture in furniture_types:
        for condition in conditions:
            class_mapping[f"{furniture}_{condition}"] = class_id
            class_id += 1
    
    print(f"Class mapping: {class_mapping}")
    
    # Process images
    source_dir = Path("furniture_dataset")
    all_images = []
    
    for furniture in furniture_types:
        for condition in conditions:
            folder_path = source_dir / furniture / condition
            if folder_path.exists():
                images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png"))
                print(f"Found {len(images)} {furniture} {condition} images")
                
                for img_path in images:
                    all_images.append({
                        'path': img_path,
                        'furniture': furniture,
                        'condition': condition,
                        'class_id': class_mapping[f"{furniture}_{condition}"]
                    })
    
    if len(all_images) == 0:
        print("❌ No images found in furniture_dataset/")
        print("Please add images to the folders as described in furniture_dataset/README.md")
        return None
    
    print(f"Total images found: {len(all_images)}")
    
    # Split into train/val (80/20)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Copy images and create labels
    for split, images in [("train", train_images), ("val", val_images)]:
        for img_data in images:
            # Copy image
            img_name = img_data['path'].name
            dest_img = dataset_dir / split / "images" / img_name
            shutil.copy2(img_data['path'], dest_img)
            
            # Create YOLO label (center of image, full size)
            img = cv2.imread(str(img_data['path']))
            if img is not None:
                h, w = img.shape[:2]
                
                # Create bounding box for entire image (assuming furniture fills most of image)
                x_center = 0.5
                y_center = 0.5
                width = 0.8  # 80% of image width
                height = 0.8  # 80% of image height
                
                # Write label file
                label_name = img_data['path'].stem + ".txt"
                label_path = dataset_dir / split / "labels" / label_name
                
                with open(label_path, 'w') as f:
                    f.write(f"{img_data['class_id']} {x_center} {y_center} {width} {height}\n")
    
    # Create YAML config
    class_names = []
    for furniture in furniture_types:
        for condition in conditions:
            class_names.append(f"{furniture}_{condition}")
    
    config = {
        'path': str(dataset_dir.absolute()),
        'train': str((dataset_dir / "train").absolute()),
        'val': str((dataset_dir / "val").absolute()),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = 'real_furniture_data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset created successfully!")
    print(f"Classes: {class_names}")
    print(f"YAML config saved to: {yaml_path}")
    
    return yaml_path

def train_model(yaml_path):
    """Train the YOLO model"""
    print("Training YOLO model...")
    
    # Load model
    model = YOLO("yolov8n.yaml")
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",
        project="real_furniture_runs",
        name="furniture_condition_detection",
        save=True,
        plots=True,
        val=True
    )
    
    print("Training completed!")
    return results

def test_model():
    """Test the trained model"""
    print("Testing trained model...")
    
    model_path = "real_furniture_runs/furniture_condition_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found!")
        return
    
    model = YOLO(model_path)
    
    # Test on sample images
    test_images = ["house_interior.jpg", "office.jpg"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting on: {img_path}")
            
            results = model(img_path, conf=0.3)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        print(f"  Found: {class_name} (confidence: {confidence:.2f})")

def main():
    """Main function"""
    print("Real Furniture Condition Detection Training")
    print("=" * 50)
    
    # Check if furniture_dataset exists
    if not os.path.exists("furniture_dataset"):
        print("❌ furniture_dataset folder not found!")
        print("Please create the folder structure as described in the README")
        return
    
    # Create dataset
    yaml_path = create_yolo_dataset()
    if yaml_path is None:
        return
    
    # Train model
    train_model(yaml_path)
    
    # Test model
    test_model()
    
    print("\n✅ Training completed!")
    print("You can now use the trained model with fix_detection.py")

if __name__ == "__main__":
    main()
