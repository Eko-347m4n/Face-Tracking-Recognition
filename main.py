import cv2
import mediapipe as mp 
import logging
import time 
from core import display 
from core import roi as roi_module 
from core import dataset_manager
from core import camera_handler
from core import processing_pipeline

# --- Configuration ---
DATASET_PATH = "images/"  # Path to the folder containing dataset images
WEBCAM_ID = 0             # Usually 0 for built-in webcam, try 1 if 0 doesn't work
RECOGNITION_TOLERANCE = 0.65 # Lower is stricter (more false negatives), higher is more lenient (more false positives)
FRAME_RESIZE_FACTOR = 0.5  # Resize frame for faster processing (e.g., 0.25 for 1/4 size).
                              # Increase to 0.5, 0.75, or 1.0 for better detection of smaller/distant faces, at the cost of performance.
FRAMES_TO_SKIP_AFTER_DETECTION = 5 # Number of frames to track before re-detecting. 0 to disable tracking.
MEDIAPIPE_MODEL_SELECTION = 0 # 0 for short-range (<=2m), 1 for full-range (potentially better for varied distances but might be slower/less accurate up close)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5 # Minimum confidence for MediaPipe face detection (0.0 - 1.0)
UNKNOWN_PERSON_LABEL = "Tidak dikenal"
WINDOW_NAME = "Video Face Recognition" # Define window name as a constant
ROI_ENABLED = True
ROI_X_FACTOR = 0.025  # ROI dimulai pada 2.5% dari kiri frame
ROI_Y_FACTOR = 0.025  # ROI dimulai pada 2.5% dari atas frame
ROI_W_FACTOR = 0.95   # Lebar ROI 95% dari lebar frame
ROI_H_FACTOR = 0.95   # Tinggi ROI 95% dari tinggi frame
QUEUE_SIZE = 2 # Max size for inter-thread queues
ADAPTIVE_ROI_PADDING = 30 # Pixels padding around detected faces for adaptive ROI

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
UNKNOWN_FACE_ID = "---" # ID untuk wajah tidak dikenal

# --- Main Application ---
def main():
    # 1. Load known faces from the dataset
    known_face_encodings, known_face_labels, known_face_ids = \
        dataset_manager.load_known_faces_from_dataset(DATASET_PATH)

    # 2. Initialize webcam
    video_capture = camera_handler.initialize_webcam(WEBCAM_ID)
    if not video_capture:
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

    # 2.5 Initialize MediaPipe Face Detection (using parameters from pipeline_config)
    mp_face_detection_solution = mp.solutions.face_detection
    mp_face_detector = mp_face_detection_solution.FaceDetection(
        model_selection=pipeline_config['MEDIAPIPE_MODEL_SELECTION'],
        min_detection_confidence=pipeline_config['MEDIAPIPE_MIN_DETECTION_CONFIDENCE']
    )

    pipeline = processing_pipeline.FaceProcessingPipeline(
        video_capture, mp_face_detector,
        known_face_encodings, known_face_labels, known_face_ids,
        pipeline_config
    )
    pipeline.start()

    logging.info("Application started. Press 'q' on the video window to quit.")

    # --- Main Display Loop ---
    fps_start_time = time.time()
    fps_frame_count = 0
    displayed_fps = 0.0
    
    # Variables to hold the latest data for display
    latest_frame_to_display = None
    latest_face_locations = []
    latest_face_details = []
    current_roi_coords_for_drawing = None # Akan menyimpan (x,y,w,h) ROI yang digunakan
    latest_total_processing_time_ms = 0.0

    quit_signal_received = False
    while not quit_signal_received:
        display_data = pipeline.get_display_data(timeout=0.01) # Short timeout for responsiveness

        if display_data:
            (frame_to_display, face_locs, face_dets, roi_coords,
             det_time, id_time) = display_data

            latest_frame_to_display = frame_to_display
            latest_face_locations = face_locs
            latest_face_details = face_dets
            current_roi_coords_for_drawing = roi_coords
            latest_total_processing_time_ms = det_time + id_time
        elif latest_frame_to_display is None:
            # No frame processed yet and no previous frame to show
            if not video_capture.isOpened() and not pipeline.stop_event.is_set(): # Check if capture might have stopped
                logging.info("Main loop: Video capture seems to have stopped early.")
                break 
            time.sleep(0.01) # Wait briefly
            continue
        
        # If display_data is None but we have a latest_frame_to_display, continue showing it
        if latest_frame_to_display is None: # Should only happen if pipeline stops before first frame
            # No new processed frame, continue displaying the last one if available
            continue

        # --- Display Logic (similar to before, but using latest_ variables) ---
        display_frame = latest_frame_to_display.copy() # Work on a copy

        # Kalkulasi FPS
        fps_frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - fps_start_time
        if elapsed_time > 1.0: # Update FPS setiap 1 detik
            displayed_fps = fps_frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_frame_count = 0 # Reset frame count for next interval
            fps_start_time = current_time

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
            1.0, # Karena face_locations sudah absolut, faktor resize untuk display adalah 1.0
            UNKNOWN_PERSON_LABEL,
            displayed_fps,      # Teruskan FPS yang dihitung
            latest_total_processing_time_ms  # Teruskan waktu pemrosesan yang dihitung
        )

        # Show the frame and check for quit command
        if display.show_frame(WINDOW_NAME, display_frame):
            logging.info("Quit signal received from display window.")
            quit_signal_received = True
            break
        
        if pipeline.stop_event.is_set() and display_data is None: # Pipeline stopped and queue is empty
            logging.info("Main loop: Pipeline has stopped and display queue is empty.")
            break

    # --- Cleanup ---
    logging.info("Main loop finished. Cleaning up threads and resources...")
    pipeline.stop() # Signal and join threads within the pipeline class
    video_capture.release()
    if mp_face_detector:
        mp_face_detector.close() # Release MediaPipe resources
    display.destroy_all_windows() # Use the display module's cleanup function
    logging.info("Application stopped and resources released.")

if __name__ == '__main__':
    main()
