import os
import cv2
import time
import threading
import json
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.abspath('uploads')
RESULT_FOLDER = os.path.abspath('static/results')
CROP_FOLDER = os.path.abspath('static/results/crops')
HISTORY_FILE = os.path.abspath('history.json')

for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CROP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load Models (Heavy)
try:
    logger.info("Loading YOLO models...")
    model_coco = YOLO("yolo26x.pt")
    model_helmet = YOLO("runs/detect/helmet_only_model_x/weights/best.pt")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    # In a real system, we might want to exit or handle this gracefully

# Global task tracking
processing_tasks = {}

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return {}

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving history: {e}")

def is_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

def process_video_async(input_path, output_name):
    global processing_tasks
    output_path = os.path.join(RESULT_FOLDER, output_name)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_path}")
        processing_tasks[output_name]["status"] = "failed"
        processing_tasks[output_name]["error"] = "Could not open video"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Encoder settings
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.warning("H264 encoder failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processing_tasks[output_name] = {
        "status": "processing", 
        "progress": 0, 
        "violators": 0, 
        "helmets": 0,
        "violator_images": []
    }

    frame_count = 0
    max_v = 0
    max_h = 0
    violator_images = []
    last_save_time = -1.0
    COOLDOWN = 2.0 # Save at most one crop every 2 seconds of video

    logger.info(f"Started processing {output_name} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        try:
            res_coco = model_coco(frame, verbose=False)[0]
            res_helmet = model_helmet(frame, verbose=False)[0]
        except Exception as e:
            logger.error(f"Inference error on frame {frame_count}: {e}")
            continue

        persons = []
        motorcycles = []
        for box in res_coco.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            if cls == 0: persons.append(coords) # person
            elif cls == 3: motorcycles.append(coords) # motorcycle

        no_helmets = []
        with_helmets = []
        for box in res_helmet.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            if cls == 0: no_helmets.append(coords) # no_helmet
            elif cls == 1: with_helmets.append(coords) # helmet

        f_v = 0
        f_h = 0
        current_time = frame_count / fps
        
        for p_box in persons:
            on_bike = any(is_inside(p_box, m_box) for m_box in motorcycles)
            if on_bike:
                has_no = any(is_inside(nh_box, p_box) for nh_box in no_helmets)
                if has_no:
                    f_v += 1
                    x1, y1, x2, y2 = map(int, p_box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "VIOLATION", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    
                    if current_time - last_save_time >= COOLDOWN and len(violator_images) < 100:
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, y2_c = min(width, x2), min(height, y2)
                        crop = frame[y1_c:y2_c, x1_c:x2_c]
                        if crop.size > 0:
                            img_id = f"crop_{int(time.time() * 1000)}_{len(violator_images)}.jpg"
                            cv2.imwrite(os.path.join(CROP_FOLDER, img_id), crop)
                            violator_images.append(img_id)
                            last_save_time = current_time
                else:
                    has_yes = any(is_inside(h_box, p_box) for h_box in with_helmets)
                    if has_yes:
                        f_h += 1
                        x1, y1, x2, y2 = map(int, p_box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "SAFE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)
        frame_count += 1
        max_v = max(max_v, f_v)
        max_h = max(max_h, f_h)

        if frame_count % 10 == 0 or frame_count == total_frames:
            processing_tasks[output_name].update({
                "progress": int((frame_count / total_frames) * 100),
                "violators": max_v,
                "helmets": max_h,
                "violator_images": violator_images
            })

    cap.release()
    out.release()
    
    logger.info(f"Completed processing {output_name}")
    processing_tasks[output_name]["status"] = "completed"
    
    # Save to history
    history = load_history()
    history[output_name] = {
        "name": output_name,
        "violators": max_v,
        "helmets": max_h,
        "violator_images": violator_images,
        "status": "completed",
        "timestamp": time.time()
    }
    save_history(history)

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = f"vid_{int(time.time())}.mp4"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(file_path)
        logger.info(f"Video uploaded: {filename}")
        
        # Initialize task before starting thread
        processing_tasks[filename] = {"status": "starting", "progress": 0}
        
        thread = threading.Thread(target=process_video_async, args=(file_path, filename))
        thread.start()
        
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status/<filename>')
def get_status(filename):
    task = processing_tasks.get(filename)
    if not task:
        history = load_history()
        task = history.get(filename)
        if not task:
            return jsonify({"status": "not_found"}), 404
    return jsonify(task)

@app.route('/api/history')
def get_history_api():
    history = load_history()
    # Sort by timestamp descending
    sorted_history = sorted(history.values(), key=lambda x: x.get('timestamp', 0), reverse=True)
    return jsonify(sorted_history)

@app.route('/api/history/<filename>', methods=['DELETE'])
def delete_item(filename):
    history = load_history()
    if filename in history:
        item = history[filename]
        # Clean up files
        try:
            for img in item.get('violator_images', []):
                p = os.path.join(CROP_FOLDER, img)
                if os.path.exists(p): os.remove(p)
            
            res_p = os.path.join(RESULT_FOLDER, filename)
            if os.path.exists(res_p): os.remove(res_p)
            
            up_p = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(up_p): os.remove(up_p)
            
            del history[filename]
            save_history(history)
            logger.info(f"Deleted history item: {filename}")
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Item not found"}), 404

@app.route('/api/download/<filename>')
def download_api(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

# Static serving
@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/crops/<path:filename>')
def serve_crops(filename):
    return send_from_directory(CROP_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
