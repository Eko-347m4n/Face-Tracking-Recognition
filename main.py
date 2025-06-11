import mediapipe as mp
import logging
import time
import os  # Added for path operations
from core import display
from core import roi as roi_module
from core import dataset_manager
from core import camera_handler
from core import processing_pipeline

# --- Configuration ---
DATASET_PATH = "images/"  # Path to the folder containing dataset images
WEBCAM_ID = 0  # Usually 0 for built-in webcam, try 1 if 0 doesn't work
RECOGNITION_TOLERANCE = 0.65  # Lower is stricter (more false negatives),
                              # higher is more lenient (more false positives)
FRAME_RESIZE_FACTOR = 0.5  # Resize frame for faster processing (e.g., 0.25).
                           # Increase for better detection of smaller/distant
                           # faces, at the cost of performance.
FRAMES_TO_SKIP_AFTER_DETECTION = 5  # Frames to track before re-detecting.
                                    # 0 to disable tracking.
MEDIAPIPE_MODEL_SELECTION = 0  # 0 for short-range (<=2m), 1 for full-range
                               # (better for varied distances, might be slower
                               # or less accurate up close)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5  # Min confidence for MediaPipe face
                                          # detection (0.0 - 1.0)
UNKNOWN_PERSON_LABEL = "Tidak dikenal"
WINDOW_NAME = "Video Face Recognition"  # Define window name as a constant
ROI_ENABLED = True
ROI_X_FACTOR = 0.025  # ROI dimulai pada 2.5% dari kiri frame
ROI_Y_FACTOR = 0.025  # ROI dimulai pada 2.5% dari atas frame
ROI_W_FACTOR = 0.95   # Lebar ROI 95% dari lebar frame
ROI_H_FACTOR = 0.95   # Tinggi ROI 95% dari tinggi frame
QUEUE_SIZE = 2  # Max size for inter-thread queues
ADAPTIVE_ROI_PADDING = 30  # Pixels padding for adaptive ROI

# New configurations for dataset auto-reloading
DATASET_AUTO_RELOAD_ENABLED = True  # Enable/disable automatic dataset reloading
DATASET_CHECK_INTERVAL_SECONDS = 30  # How often to check for dataset changes

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Constants ---
UNKNOWN_FACE_ID = "---"  # ID untuk wajah tidak dikenal


# --- Helper Function for Dataset Snapshot ---
def get_dataset_snapshot(dataset_path):
    """
    Gets a set of image file paths in the dataset directory.
    Used to detect changes in the dataset.
    """
    image_files = []
    if not os.path.isdir(dataset_path):
        logging.warning(
            f"Dataset path '{dataset_path}' does not exist or is not a directory."
        )
        return set()
    for root, _, files in os.walk(dataset_path):
        for file in files:
            # Consider common image extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(os.path.join(root, file))
    return set(sorted(image_files))  # Sort for consistent set comparison


# --- Main Application ---
def main():
    # 1. Load known faces from the dataset
    logging.info(f"Loading initial dataset from: {DATASET_PATH}")
    known_face_encodings, known_face_labels, known_face_ids = \
        dataset_manager.load_known_faces_from_dataset(DATASET_PATH)

    # 2. Initialize webcam
    video_capture = camera_handler.initialize_webcam(WEBCAM_ID)
    if not video_capture:
        logging.error("Failed to initialize webcam. Exiting application.")
        return

    # 3. Configure and Initialize Processing Pipeline
    pipeline_config = {
        'ROI_ENABLED': ROI_ENABLED,
        'ROI_PARAMS': {
            'x_factor': ROI_X_FACTOR,
            'y_factor': ROI_Y_FACTOR,
            'w_factor': ROI_W_FACTOR,
            'h_factor': ROI_H_FACTOR,
        },
        'FRAME_RESIZE_FACTOR': FRAME_RESIZE_FACTOR,
        'RECOGNITION_TOLERANCE': RECOGNITION_TOLERANCE,
        'FRAMES_TO_SKIP_AFTER_DETECTION': FRAMES_TO_SKIP_AFTER_DETECTION,
        'UNKNOWN_PERSON_LABEL': UNKNOWN_PERSON_LABEL,
        'UNKNOWN_FACE_ID': UNKNOWN_FACE_ID,
        'QUEUE_SIZE': QUEUE_SIZE,
        'ADAPTIVE_ROI_PADDING': ADAPTIVE_ROI_PADDING,
        'MEDIAPIPE_MODEL_SELECTION': MEDIAPIPE_MODEL_SELECTION,
        'MEDIAPIPE_MIN_DETECTION_CONFIDENCE': MEDIAPIPE_MIN_DETECTION_CONFIDENCE
    }

    # For dataset auto-reloading
    current_dataset_snapshot = None
    last_dataset_check_time = 0
    if DATASET_AUTO_RELOAD_ENABLED:
        current_dataset_snapshot = get_dataset_snapshot(DATASET_PATH)
        last_dataset_check_time = time.time()
    if not known_face_encodings:
        logging.warning(
            "Initial dataset load resulted in zero known faces. "
            "Recognition will be limited."
        )
    else:
        logging.info(
            f"Initial dataset loaded. Found {len(known_face_encodings)} known faces."
        )
    # Initialize MediaPipe Face Detection (reused across pipeline restarts)
    mp_face_detection_solution = mp.solutions.face_detection
    mp_face_detector = mp_face_detection_solution.FaceDetection(
        model_selection=pipeline_config['MEDIAPIPE_MODEL_SELECTION'],
        min_detection_confidence=pipeline_config['MEDIAPIPE_MIN_DETECTION_CONFIDENCE']
    )

    # Initialize and start the processing pipeline
    active_pipeline = processing_pipeline.FaceProcessingPipeline(
        video_capture, mp_face_detector,
        known_face_encodings, known_face_labels, known_face_ids,
        pipeline_config
    )
    active_pipeline.start()

    logging.info("Application started. Press 'q' on the video window to quit.")

    # --- Main Display Loop ---
    fps_start_time = time.time()
    fps_frame_count = 0
    displayed_fps = 0.0

    # Variables to hold the latest data for display
    latest_frame_to_display = None
    latest_face_locations = []
    latest_face_details = []
    current_roi_coords_for_drawing = None  # Stores (x,y,w,h) of ROI used
    latest_total_processing_time_ms = 0.0
    quit_signal_received = False
    while not quit_signal_received:
        current_time_loop = time.time()

        # --- Dataset Auto-Reload Logic ---
        if DATASET_AUTO_RELOAD_ENABLED and \
           (current_time_loop - last_dataset_check_time > DATASET_CHECK_INTERVAL_SECONDS):
            logging.debug(f"Checking for dataset changes in '{DATASET_PATH}'...")
            new_snapshot = get_dataset_snapshot(DATASET_PATH)

            if new_snapshot != current_dataset_snapshot:
                logging.info(
                    "Dataset change detected. Reloading known faces and "
                    "restarting pipeline..."
                )

                if active_pipeline:
                    active_pipeline.stop()  # Signal and wait for threads to join
                    logging.info("Old pipeline stopped.")

                # --- Force re-scan by deleting the pickle file ---
                pickle_file_path = os.path.join(
                    DATASET_PATH, "dataset_encodings.pkl"
                )
                if os.path.exists(pickle_file_path):
                    try:
                        os.remove(pickle_file_path)
                        logging.info(
                            f"Removed '{pickle_file_path}' to force dataset "
                            "re-scan upon change detection."
                        )
                    except OSError as e:
                        logging.warning(
                            f"Could not remove pickle file '{pickle_file_path}' "
                            f"for re-scan: {e}"
                        )
                # --- End force re-scan ---

                # Reload known faces
                # dataset_manager will now re-process images as pickle is removed
                known_face_encodings, known_face_labels, known_face_ids = \
                    dataset_manager.load_known_faces_from_dataset(DATASET_PATH)

                if not known_face_encodings:
                    logging.warning(
                        "Dataset reload resulted in zero known faces. "
                        "Recognition will be limited."
                    )
                else:
                    logging.info(
                        f"Dataset reloaded. Found {len(known_face_encodings)} "
                        "known faces."
                    )
                current_dataset_snapshot = new_snapshot
                # Re-initialize and start a new pipeline
                active_pipeline = processing_pipeline.FaceProcessingPipeline(
                    video_capture, mp_face_detector,  # Reuse video_capture and mp_face_detector
                    known_face_encodings, known_face_labels, known_face_ids,
                    pipeline_config
                )
                active_pipeline.start()
                logging.info("New pipeline started with updated dataset.")

                # Reset display variables to wait for new pipeline's output
                latest_frame_to_display = None
                latest_face_locations = []
                latest_face_details = []
                current_roi_coords_for_drawing = None
                latest_total_processing_time_ms = 0.0
                # Reset FPS counter
                fps_start_time = time.time()
                fps_frame_count = 0
                displayed_fps = 0.0

            last_dataset_check_time = current_time_loop
        # --- End of Dataset Auto-Reload Logic ---

        # Attempt to get new data from the pipeline
        new_display_data = None
        if active_pipeline and not active_pipeline.stop_event.is_set():
            new_display_data = active_pipeline.get_display_data(timeout=0.01)  # Short timeout

        if new_display_data:
            (frame_to_display, face_locs, face_dets, liveness_statuses, roi_coords,
             det_time, id_time) = new_display_data

            latest_frame_to_display = frame_to_display
            latest_face_locations = face_locs
            latest_face_details = face_dets
            latest_liveness_statuses = liveness_statuses
            current_roi_coords_for_drawing = roi_coords
            latest_total_processing_time_ms = det_time + id_time
        elif latest_frame_to_display is None:  # No new data AND no previous frame
            if video_capture and not video_capture.isOpened():
                # If pipeline isn't already stopping
                if not active_pipeline or not active_pipeline.stop_event.is_set():
                    logging.error("Video capture is not open. Stopping application.")
                    quit_signal_received = True
                    continue

            logging.debug("Waiting for frame from pipeline...")
            time.sleep(0.05)  # Wait a bit for the pipeline to produce a frame
            continue

        # If latest_frame_to_display is still None here, it means we are waiting (e.g. after reload)
        if latest_frame_to_display is None:
            time.sleep(0.05)  # Continue waiting
            continue

        # --- Display Logic (similar to before, but using latest_ variables) ---
        display_frame = latest_frame_to_display.copy() # Work on a copy

        # Kalkulasi FPS
        fps_frame_count += 1
        elapsed_time_fps = current_time_loop - fps_start_time
        if elapsed_time_fps > 1.0:
            displayed_fps = fps_frame_count / elapsed_time_fps if elapsed_time_fps > 0 else 0
            fps_frame_count = 0  # Reset frame count for next interval
            fps_start_time = current_time_loop

        # Gambar batas ROI pada frame jika diaktifkan dan koordinat tersedia
        if ROI_ENABLED and current_roi_coords_for_drawing:
            frame_h_disp, frame_w_disp = display_frame.shape[:2]
            # Hanya gambar kotak ROI visual jika tidak mewakili seluruh frame
            if not (current_roi_coords_for_drawing[0] == 0 and
                    current_roi_coords_for_drawing[1] == 0 and
                    current_roi_coords_for_drawing[2] == frame_w_disp and
                    current_roi_coords_for_drawing[3] == frame_h_disp):
                roi_module.draw_roi_boundary(display_frame, current_roi_coords_for_drawing)

        display.draw_detections_on_frame(
            display_frame,
            latest_face_locations,
            latest_face_details,
            latest_liveness_statuses,
            1.0,  # face_locations are absolute, so display resize factor is 1.0
            UNKNOWN_PERSON_LABEL,
            displayed_fps,
            latest_total_processing_time_ms
        )

        # Show the frame and check for quit command
        if display.show_frame(WINDOW_NAME, display_frame):
            logging.info("Quit signal received from display window.")
            quit_signal_received = True

        # If pipeline stopped for reasons other than quit signal (e.g. video file ended)
        if (active_pipeline and
                active_pipeline.stop_event.is_set() and
                not new_display_data and
                not quit_signal_received):
            # This might indicate the source ended or an issue.
            # Check if it's not immediately after a reload attempt.
            is_during_reload_transition = DATASET_AUTO_RELOAD_ENABLED and \
                (time.time() - last_dataset_check_time < 2.0)  # Small window

            if not is_during_reload_transition:
                if video_capture and not video_capture.isOpened():
                    logging.info(
                        "Pipeline stopped and video capture is closed. "
                        "Assuming end of source or error."
                    )
                else:
                    logging.warning(
                        "Pipeline stopped unexpectedly. Assuming end of source or error."
                    )
                quit_signal_received = True  # Trigger shutdown

    # --- Cleanup ---
    logging.info("Main loop finished. Cleaning up threads and resources...")
    if active_pipeline:
        active_pipeline.stop()
    if video_capture:
        video_capture.release()
    if mp_face_detector:
        mp_face_detector.close()  # Release MediaPipe resources
    display.destroy_all_windows()  # Use the display module's cleanup function
    logging.info("Application stopped and resources released.")

if __name__ == '__main__':
    main()
