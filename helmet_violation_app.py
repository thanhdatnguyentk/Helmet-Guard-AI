import cv2
from ultralytics import YOLO
import numpy as np

def is_inside(inner_box, outer_box):
    """Check if the center of inner_box is inside outer_box."""
    # box format: [x1, y1, x2, y2]
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    # Center of inner box
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

def main():
    # Load Models
    # model: COCO model for detecting persons and motorcycles
    # model_helmet: Specialized model for detecting 'With Helmet' (cls 1) and 'Without Helmet' (cls 0)
    model_coco = YOLO("yolo26x.pt")
    model_helmet = YOLO("runs/detect/helmet_only_model_x/weights/best.pt")

    cap = cv2.VideoCapture("data/video3.mp4")
    
    # Get video properties for scaling UI if needed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detections
        results_coco = model_coco(frame, verbose=False)[0]
        results_helmet = model_helmet(frame, verbose=False)[0]

        # Extract relevant COCO classes: person=0, motorcycle=3
        persons = []
        motorcycles = []
        for box in results_coco.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            if cls == 0: # person
                persons.append(coords)
            elif cls == 3: # motorcycle
                motorcycles.append(coords)

        # Extract Helmet classes: 0: Without Helmet, 1: With Helmet
        no_helmets = []
        with_helmets = []
        for box in results_helmet.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            if cls == 0: # Without Helmet
                no_helmets.append(coords)
            elif cls == 1: # With Helmet
                with_helmets.append(coords)

        # Logic: A person is "violating" if:
        # 1. They are on a motorcycle (overlap between person and motorcycle)
        # 2. They have a "no helmet" detection near/on them.
        
        violators_count = 0
        active_violators_coords = []
        non_violators_coords = []

        for p_box in persons:
            # Check if person is on a motorcycle
            on_bike = any(is_inside(p_box, m_box) for m_box in motorcycles)
            
            if on_bike:
                # Check if this person has a helmet or not
                # We find the 'no helmet' detection that is inside the person's bounding box
                has_no_helmet = any(is_inside(nh_box, p_box) for nh_box in no_helmets)
                
                if has_no_helmet:
                    violators_count += 1
                    active_violators_coords.append(p_box)
                else:
                    non_violators_coords.append(p_box)

        # Draw UI
        # First, plot coco detections for context (optional, cleaner without)
        display_frame = frame.copy()
        # Highlight non-violators
        for v_box in non_violators_coords:
            x1, y1, x2, y2 = map(int, v_box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(display_frame, "WITH HELMET", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Highlight violators
        for v_box in active_violators_coords:
            x1, y1, x2, y2 = map(int, v_box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display_frame, "NO HELMET", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Overlay summary
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        cv2.putText(display_frame, f"Violators (No Helmet): {violators_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Helmet Violation Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
