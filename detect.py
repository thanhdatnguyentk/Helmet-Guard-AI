from ultralytics import YOLO
import cv2
import torch

# Load models
model = YOLO("yolo26x.pt")
model_helmet = YOLO("runs/detect/helmet_only_model_x/weights/best.pt")

cap = cv2.VideoCapture("data/video1.mp4")

# Run inference with both models
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detections
    results_coco = model(frame)
    results_helmet = model_helmet(frame)
    
    # Plot first model (COCO)
    plot_frame = results_coco[0].plot()
    
    # Plot second model (Helmet) onto the same frame
    final_frame = results_helmet[0].plot(img=plot_frame)
    
    # Display the results
    cv2.imshow("Combined Detections (COCO + Helmet)", final_frame) 
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
