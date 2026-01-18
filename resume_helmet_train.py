from ultralytics import YOLO
import os

def main():
    checkpoint_path = "runs/detect/helmet_only_model_x/weights/last.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Resuming training from {checkpoint_path}...")
    
    # Initialize from the last checkpoint
    model = YOLO(checkpoint_path)
    
    # Resume training
    # Note: resume=True will automatically load the arguments from the checkpoint's args.yaml
    model.train(resume=True)

if __name__ == "__main__":
    main()
