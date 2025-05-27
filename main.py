import cv2
import face_recognition
import logging
import os
import pickle
import time 
from core import utils
from core import recognizer
from core import display 

# --- Configuration ---
DATASET_PATH = "images/"  # Path to the folder containing dataset images
WEBCAM_ID = 0             # Usually 0 for built-in webcam, try 1 if 0 doesn't work
RECOGNITION_TOLERANCE = 0.6 # Lower is stricter (more false negatives), higher is more lenient (more false positives)
FRAME_RESIZE_FACTOR = 0.25  # Resize frame for faster processing (e.g., 0.25 for 1/4 size). Use 1.0 for original size.
FRAMES_TO_SKIP_AFTER_DETECTION = 2 # Proses setiap frame untuk sensitivitas maksimal terhadap gerakan.
UNKNOWN_PERSON_LABEL = "Tidak dikenal"
WINDOW_NAME = "Video Face Recognition" # Define window name as a constant

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables for Known Faces ---
# These will be populated by load_known_faces()
# Pertimbangkan untuk merangkum ini dalam sebuah kelas atau mengembalikannya dari fungsi
known_face_encodings = []
known_face_labels = []
known_face_ids = [] # Tambahkan list untuk ID unik
UNKNOWN_FACE_ID = "---" # ID untuk wajah tidak dikenal

def load_known_faces(dataset_path):
    global known_face_encodings, known_face_labels, known_face_ids
    # Initialize global lists to ensure they are clean for this run
    known_face_encodings = []
    known_face_labels = []
    known_face_ids = []

    pickle_file_name = "dataset_encodings.pkl"
    pickle_file_path = os.path.join(dataset_path, pickle_file_name)

    # 1. Attempt to load from pickle file
    if os.path.exists(pickle_file_path):
        logging.info(f"Attempting to load known faces from '{pickle_file_path}'...")
        try:
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
                if 'encodings' in data and 'labels' in data and 'ids' in data and \
                   data['encodings'] and data['labels'] and data['ids'] and \
                   len(data['encodings']) == len(data['labels']) == len(data['ids']):
                    known_face_encodings = data['encodings']
                    known_face_labels = data['labels']
                    known_face_ids = data['ids']
                    unique_labels_with_ids = {f"{label} (ID: {id_val})" for label, id_val in zip(known_face_labels, known_face_ids)}
                    logging.info(f"Successfully loaded {len(known_face_encodings)} known face(s) from pickle: {list(unique_labels_with_ids)}")
                    return # Successfully loaded from pickle
                else:
                    logging.warning(f"Pickle file '{pickle_file_path}' is empty or malformed. Will re-encode dataset.")
        except Exception as e:
            logging.warning(f"Could not load from pickle file '{pickle_file_path}' (error: {e}). Will re-encode dataset.")
            # Attempt to remove corrupted pickle file
            try:
                os.remove(pickle_file_path)
                logging.info(f"Removed potentially corrupted pickle file: '{pickle_file_path}'.")
            except OSError as oe:
                logging.warning(f"Could not remove pickle file '{pickle_file_path}': {oe}")

    # 2. If pickle loading failed or file didn't exist, load from images and generate encodings
    logging.info(f"Loading known faces from image dataset '{dataset_path}' and generating new encodings...")
    loaded_images = utils.load_images_from_folder(dataset_path)

    if not loaded_images:
        logging.warning(f"No images found in '{dataset_path}'. The system will not recognize anyone.")
        # If a pickle file existed but was problematic (corrupted/empty), and now no images are found,
        # remove the problematic pickle file to prevent issues on next run.
        if os.path.exists(pickle_file_path):
            try:
                logging.info(f"No images found for re-encoding. Removing existing pickle file '{pickle_file_path}'.")
                os.remove(pickle_file_path)
            except OSError as e:
                logging.warning(f"Could not remove existing pickle file '{pickle_file_path}': {e}")
        return

    temp_encodings = []
    temp_labels = []
    temp_ids = []
    id_counter = 0
    for img_path, image_data in loaded_images:
        encoding = utils.encode_face(image_data)
        if encoding is not None:
            temp_encodings.append(encoding)
            label = utils.extract_label_from_filename(img_path)
            temp_labels.append(label)
            temp_ids.append(f"fid_{id_counter}")
            id_counter += 1
        else:
            logging.warning(f"No face found or could not encode face in {img_path}.")

    if temp_encodings:
        known_face_encodings = temp_encodings
        known_face_labels = temp_labels
        known_face_ids = temp_ids
        unique_labels_with_ids = {f"{label} (ID: {id_val})" for label, id_val in zip(known_face_labels, known_face_ids)}
        logging.info(f"Successfully encoded {len(known_face_encodings)} known face(s) from images: {list(unique_labels_with_ids)}")

        # 3. Save the new encodings to the pickle file
        try:
            os.makedirs(dataset_path, exist_ok=True) # Ensure dataset directory exists
            with open(pickle_file_path, 'wb') as f:
                pickle.dump({
                    'encodings': known_face_encodings,
                    'labels': known_face_labels,
                    'ids': known_face_ids
                }, f)
            logging.info(f"Encodings saved successfully to '{pickle_file_path}'.")
        except Exception as e:
            logging.error(f"Failed to save encodings to pickle file '{pickle_file_path}': {e}")
    else:
        logging.warning("No faces were encoded from the dataset. No new pickle file created/updated.")
        # If a pickle file existed (e.g. from a previous run or was corrupted) and no new encodings were generated, remove it.
        if os.path.exists(pickle_file_path):
            try:
                logging.info(f"No new encodings generated. Removing existing pickle file '{pickle_file_path}'.")
                os.remove(pickle_file_path)
            except OSError as e:
                logging.warning(f"Could not remove existing pickle file '{pickle_file_path}': {e}")

def initialize_webcam(webcam_id: int):
    """Initializes and returns the webcam capture object."""
    video_capture = cv2.VideoCapture(webcam_id)
    if not video_capture.isOpened():
        logging.error(f"Could not open webcam ID {webcam_id}. Please check connectivity.")
        return None
    return video_capture

def process_frame(frame, known_face_encodings, known_face_labels, known_face_ids, tolerance):
    # Resize frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    current_face_locations = face_recognition.face_locations(rgb_small_frame)
    current_face_encodings = face_recognition.face_encodings(rgb_small_frame, current_face_locations)

    recognized_face_details = [] # Akan menyimpan tuple (nama, id)
    for face_encoding in current_face_encodings:
        # Diasumsikan recognizer.recognize_face sekarang mengembalikan (nama, id_wajah)
        name, face_id = recognizer.recognize_face(
            known_face_encodings,
            known_face_labels,
            known_face_ids, # Argumen baru
            face_encoding,
            tolerance=tolerance,
            default_name=UNKNOWN_PERSON_LABEL,
            default_id=UNKNOWN_FACE_ID # ID default untuk tidak dikenal
        )
        recognized_face_details.append((name, face_id))
    return current_face_locations, recognized_face_details

def main():
    # 1. Load known faces from the dataset
    load_known_faces(DATASET_PATH)

    # 2. Initialize webcam
    video_capture = initialize_webcam(WEBCAM_ID)
    if not video_capture:
        return

    logging.info("Starting webcam... Press 'q' on the video window to quit.")

    # Variables for processing optimization
    face_locations = []
    face_details = [] # Akan menyimpan list dari tuple (nama, id)
    
    # Variabel untuk FPS dan Waktu Pemrosesan
    fps_start_time = time.time()
    fps_frame_count = 0
    displayed_fps = 0.0
    processing_time_ms = 0.0 # Waktu untuk process_frame
    frames_to_skip_counter = 0

    while True:
        # 3. Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to grab frame from webcam. Exiting.")
            break

        # 4. Process frame (detect and recognize faces) only if not skipping
        if frames_to_skip_counter <= 0:
            start_process_time = time.time() # Catat waktu mulai pemrosesan
            current_locations, current_details = process_frame(
                frame,
                known_face_encodings,
                known_face_labels,
                known_face_ids, # Teruskan IDs
                RECOGNITION_TOLERANCE
            )
            end_process_time = time.time() # Catat waktu selesai pemrosesan
            processing_time_ms = (end_process_time - start_process_time) * 1000

            # Update if new faces are detected or locations changed significantly (simplification: always update if processed)
            if current_locations: # or significant_change(current_locations, face_locations):
                face_locations = current_locations
                face_details = current_details
                frames_to_skip_counter = FRAMES_TO_SKIP_AFTER_DETECTION # Reset skip counter
            else:
                # Jika tidak ada wajah terdeteksi oleh process_frame, kosongkan list
                face_locations = []
                face_details = []
                frames_to_skip_counter = 0
        else:
            frames_to_skip_counter -= 1

        # Kalkulasi FPS
        fps_frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - fps_start_time

        if elapsed_time > 1.0: # Update FPS setiap 1 detik
            displayed_fps = fps_frame_count / elapsed_time
            fps_start_time = current_time
            fps_frame_count = 0

        # 5. Display the results
        # Draw detections on the current frame (modifies frame in-place)
        # Indikator performa (FPS, waktu proses) akan selalu digambar oleh display.py
        # Kotak wajah akan digambar jika face_locations tidak kosong
        display.draw_detections_on_frame(
            frame,
            face_locations, # Bisa kosong jika tidak ada wajah
            face_details,   # Bisa kosong jika tidak ada wajah
            FRAME_RESIZE_FACTOR,
            UNKNOWN_PERSON_LABEL,
            displayed_fps,      # Teruskan FPS yang dihitung
            processing_time_ms  # Teruskan waktu pemrosesan yang dihitung
        )

        # Show the frame and check for quit command
        if display.show_frame(WINDOW_NAME, frame):
            break

    # Release handle to the webcam
    video_capture.release()
    display.destroy_all_windows() # Use the display module's cleanup function
    logging.info("Webcam stopped and resources released.")

if __name__ == '__main__':
    main()