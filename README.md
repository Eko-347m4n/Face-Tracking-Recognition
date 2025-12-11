# Face Tracking & Recognition System with Liveness Detection

A robust, real-time face recognition application engineered for performance and security. This system utilizes a multi-threaded architecture to perform face detection, tracking, and liveness verification (anti-spoofing) seamlessly on standard CPU hardware.

## 🟦 1. Tech Stack & Rationale

We chose a specific stack to balance high-performance real-time processing with ease of maintenance:

*   **Core Vision: OpenCV**
    *   *Why:* Industry standard for I/O and lightweight object tracking (`TrackerCSRT`).
*   **Face Detection: MediaPipe**
    *   *Why:* Superior CPU performance compared to HOG/CNN dlib models, essential for maintaining high FPS without a GPU.
*   **Recognition: face_recognition (dlib wrapper)**
    *   *Why:* Provides state-of-the-art accuracy (99.38%) with a clean, pythonic API for 128-d face encoding.
*   **Anti-Spoofing: dlib (68 landmarks)**
    *   *Why:* Required for precise facial landmark detection to calculate the Eye Aspect Ratio (EAR) for liveness validation.

## 🟦 2. Architecture & Design

The project follows a **Modular Monolith** structure to separate concerns effectively:

*   **`main.py` (Orchestrator):** Handles configuration, lifecycle management, and graceful shutdowns.
*   **`core/` (Service Layer):**
    *   `processing_pipeline.py`: The central controller managing the multi-threaded data flow.
    *   `recognizer.py` & `anti_spoofing.py`: Pure domain logic modules.
    *   `dataset_manager.py`: Data access layer handling caching and storage.
    *   `display.py`: Presentation layer responsible for rendering the UI.

## 🟦 3. Key Features & Business Logic

### 🚀 Detect-then-Track Strategy
To solve the latency problem inherent in face detection, we implement a hybrid approach:
1.  **Expensive Detection:** `MediaPipe` is run periodically or when tracking fails.
2.  **Cheap Tracking:** Between detections, `OpenCV Tracker` follows the object, drastically reducing CPU load.
3.  **Adaptive ROI:** Search areas are dynamically reduced based on the last known face position.

### 🛡️ Anti-Spoofing (Liveness Check)
Security is paramount. The system validates that a detected face is "live" and not a static photograph:
*   **Mechanism:** Uses `dlib` to predict 68 facial landmarks.
*   **Algorithm:** Calculates the **Eye Aspect Ratio (EAR)** to detect natural blinking patterns. If no blink is detected over a set period, access is denied.

### ⚡ Smart Caching
Startup time is optimized by caching processed face encodings:
*   **Logic:** `dataset_manager.py` checks file modification times.
*   **Result:** If the dataset hasn't changed, encodings are loaded from a `.pkl` file in milliseconds, skipping the expensive re-encoding process.

## 🟦 4. System Data Flow

The application uses a **Multi-threaded Producer-Consumer** architecture to prevent UI blocking:

1.  **Capture Thread:** Fetches raw frames from the camera $\rightarrow$ `frame_queue`.
2.  **Detection Thread:** Consumes frames $\rightarrow$ Runs Detect/Track logic $\rightarrow$ Pushes results to `face_queue`.
3.  **Identification Thread:** Consumes ROIs $\rightarrow$ Performs Encoding & Liveness Checks $\rightarrow$ Updates state.
4.  **Main Thread:** Aggregates all data $\rightarrow$ Renders the final UI via `display.py`.

## 📂 Project Structure

```text
/
├── core/
│   ├── anti_spoofing.py      # Liveness detection (blink analysis)
│   ├── camera_handler.py     # Threaded camera I/O
│   ├── dataset_manager.py    # Caching and dataset loading
│   ├── display.py            # UI rendering (OpenCV drawing)
│   ├── processing_pipeline.py# Core multi-threading logic
│   ├── recognizer.py         # Face matching logic
│   ├── roi.py                # Region of Interest utilities
│   └── utils.py              # General helpers
├── images/                   # Dataset directory
├── main.py                   # Entry point
└── requirements.txt          # Project dependencies
```

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.8+
*   Webcam

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Face-Tracking-Recognition
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install opencv-python opencv-contrib-python dlib face_recognition mediapipe numpy
    ```
    *Note: Installing `dlib` may require CMake installed on your system.*

4.  **Prepare Dataset:**
    *   Put images of known people in the `images/` folder.
    *   Filenames should be `Name_Identifier.jpg` (e.g., `Elon_1.jpg`). The system automatically extracts "Elon" as the label.

## 🚀 Usage

Run the main application:
```bash
python main.py
```

*   **Green Box:** Recognized & Live user.
*   **Red Box:** Unknown user or potential spoof attempt.
*   **Stats:** FPS and processing status are displayed in real-time.
*   **Quit:** Press `q` to exit.