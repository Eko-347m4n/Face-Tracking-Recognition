# Real-Time Face Recognition System

A Python-based real-time face recognition system that uses a webcam to detect and identify known faces from a dataset.

## Features

*   **Dataset Loading:** Dynamically loads and encodes faces from an image dataset.
*   **Real-Time Detection:** Captures video from a webcam and detects faces in real-time.
*   **Face Recognition:** Compares detected faces against the known dataset.
*   **Visual Feedback:**
    *   Draws bounding boxes around detected faces.
    *   Labels faces with their recognized name or "Tidak dikenal" (Unknown).
    *   Bounding box color changes based on recognition status (Green for known, Red for unknown).
*   **Performance Optimization:** Includes options for frame resizing and smart frame skipping to balance performance and responsiveness.

## Requirements

*   Python 3.7+
*   OpenCV (`opencv-python`)
*   face_recognition
*   NumPy

## Setup & Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install opencv-python face_recognition numpy
    ```

4.  **Prepare the Dataset:**
    *   Create a folder named `images/` in the project's root directory.
    *   Place your dataset images (e.g., `.jpg`, `.png`) inside the `images/` folder.
    *   Name your image files using the convention: `label_number.extension` (e.g., `john_1.jpg`, `jane_doe_1.png`, `john_2.jpg`). The part before the first underscore will be used as the person's name.

## How to Run

Navigate to the project's root directory in your terminal and run:

```bash
python main.py
```

A window will pop up showing your webcam feed.
*   Known faces will be highlighted with a **green** box and their name.
*   Unknown faces will be highlighted with a **red** box and labeled "Tidak dikenal".

Press 'q' while the video window is active to quit the application.

## Project Structure

```
├── core/
│   ├── __init__.py
│   ├── utils.py         # Utility functions for image loading, encoding, label extraction
│   └── recognizer.py    # Logic for comparing face encodings
├── images/              # Folder for dataset images (ignored by Git)
├── main.py              # Main script to run the application
├── .gitignore           # Specifies intentionally untracked files
└── README.md            # This file
```

## Configuration (in `main.py`)

You can adjust the following parameters at the top of `main.py`:

*   `DATASET_PATH`: Path to the dataset images folder (default: `"images/"`).
*   `WEBCAM_ID`: Webcam device ID (default: `0`).
*   `RECOGNITION_TOLERANCE`: How strict the face matching is (default: `0.6`). Lower is stricter.
*   `FRAME_RESIZE_FACTOR`: Factor to resize webcam frames for faster processing (default: `0.25`). `1.0` for original size.
*   `FRAMES_TO_SKIP_AFTER_DETECTION`: Number of frames to skip processing after a face is detected, to balance performance and responsiveness (default: `2`). Set to `0` to process every frame for maximum sensitivity to movement.

## Notes

*   The `images/` directory is included in `.gitignore` and will not be tracked by Git. This is to prevent large image datasets from being committed to the repository.
*   Performance can vary depending on your CPU and webcam resolution. Adjust `FRAME_RESIZE_FACTOR` and `FRAMES_TO_SKIP_AFTER_DETECTION` for optimal performance on your system.

```
