import os
import shutil
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm
import random

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def prepare_dataset():
    img_dir = "data/images"
    ann_dir = "data/annotations"
    output_base = "helmet_only_dataset"
    
    # Classes: 0: Without Helmet, 1: With Helmet (Matches existing yaml)
    label_map = {"no_helmet": 0, "helmet": 1, "with helmet": 1, "without helmet": 0}
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)
    
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
        print(f"Processing {split} split...")
        for img_name in tqdm(img_list):
            img_path = os.path.join(img_dir, img_name)
            ann_name = os.path.splitext(img_name)[0] + ".xml"
            ann_path = os.path.join(ann_dir, ann_name)
            
            if not os.path.exists(ann_path):
                continue
                
            shutil.copy(img_path, os.path.join(output_base, split, "images", img_name))
            
            tree = ET.parse(ann_path)
            root = tree.getroot()
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            
            yolo_labels = []
            for obj in root.iter("object"):
                name = obj.find("name").text.lower()
                if name in label_map:
                    cls_id = label_map[name]
                    xmlbox = obj.find("bndbox")
                    b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text), 
                         float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
                    bb = convert_box((w, h), (b[0], b[1], b[2], b[3]))
                    yolo_labels.append(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}")
            
            lbl_path = os.path.join(output_base, split, "labels", os.path.splitext(img_name)[0] + ".txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_labels))
                
    yaml_content = f"""
path: {os.path.abspath(output_base)}
train: train/images
val: val/images

names:
  0: Without Helmet
  1: With Helmet
"""
    with open("helmet_only.yaml", "w") as f:
        f.write(yaml_content)
    print("Dataset preparation complete.")

def main():
    if not os.path.exists("helmet_only_dataset"):
        prepare_dataset()
        
    model = YOLO("yolo26x.pt") # Use small model for helmet pre-training
    model.train(
        data="helmet_only.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        name="helmet_only_model_x",
        device=0,
        workers=0
    )
    
    # Save the best model to root for Stage 2
    best_path = "runs/detect/helmet_only_model/weights/best.pt"
    if os.path.exists(best_path):
        shutil.copy(best_path, "helmet_model_best.pt")
        print("Helmet model training complete. Best model saved as 'helmet_model_best.pt'")

if __name__ == "__main__":
    main()
