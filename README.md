# Helmet Guard AI 🛡️

Helmet Guard AI is an intelligent traffic safety monitoring system designed to detect and record motorcycle riders who are not wearing helmets. It features a robust **Client-Server architecture** with asynchronous video processing for smooth performance.

## ✨ Key Features

- **Dual-Model Inference**: Combines a COCO-pretrained model (for person and motorcycle detection) with a specialized student model (for helmet detection).
- **Asynchronous Processing**: Videos are processed in the background on the server, ensuring the web interface remains responsive.
- **Evidence Bench**: Automatically crops and saves snapshots of violators as they are detected.
- **Persistent History**: Keeps track of all previous analysis results, including violation counts and evidence snapshots.
- **High-Quality Playback**: Pre-renders processed videos using modern codecs to ensure smooth (24+ FPS) playback on the web.
- **Data Management**: Easily download processed results or delete old records to free up space.

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- NVIDIA GPU (Recommended for faster processing)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd no_helmet
    ```

2.  **Install dependencies**:
    ```bash
    py -3.10 -m pip install -r requirements.txt
    # Alternatively, manually install key packages:
    # py -3.10 -m pip install ultralytics flask flask-cors opencv-python
    ```

3.  **Ensure Model Weights**:
    Place `yolo26x.pt` and `runs/detect/helmet_only_model_x/weights/best.pt` in the correct directories as referenced in `app.py`.

### Running the Application

1.  **Start the Server**:
    ```powershell
    py -3.10 app.py
    ```

2.  **Access the Web App**:
    Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

## 📂 Project Structure

```text
no_helmet/
├── app.py              # Flask Backend API
├── static/             # Frontend Assets
│   ├── css/            # Modern Glassmorphism Styles
│   ├── js/             # Modular Frontend Logic
│   ├── results/        # Processed Video Output
│   └── images/         # UI Graphics
├── templates/          # HTML Templates
├── uploads/            # Temporary Video Storage
├── history.json        # Persistent Analysis Records
├── requirements.txt    # Project Dependencies
└── .gitignore          # Version Control Exclusions
```

## 🛠️ Technology Stack

- **Backend**: Python, Flask, OpenCV
- **AI/ML**: Ultralytics YOLOv11 (v26x)
- **Frontend**: Vanilla JS, CSS (Modern UI), HTML5
- **Icons**: FontAwesome

## 📝 License

This project is for educational and safety monitoring purposes.
