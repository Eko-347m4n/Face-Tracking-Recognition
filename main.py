import cv2
import face_recognition
from core import utils
from core import recognizer

# --- Configuration ---
DATASET_PATH = "images/"  # Path to the folder containing dataset images
WEBCAM_ID = 0             # Usually 0 for built-in webcam, try 1 if 0 doesn't work
RECOGNITION_TOLERANCE = 0.6 # Lower is stricter (more false negatives), higher is more lenient (more false positives)
FRAME_RESIZE_FACTOR = 0.25  # Resize frame for faster processing (e.g., 0.25 for 1/4 size). Use 1.0 for original size.
FRAMES_TO_SKIP_AFTER_DETECTION = 2 # Proses setiap frame untuk sensitivitas maksimal terhadap gerakan.

# --- Global Variables for Known Faces ---
# These will be populated by load_known_faces()
known_face_encodings = []
known_face_labels = []

def load_known_faces(dataset_path):
    global known_face_encodings, known_face_labels
    known_face_encodings = []
    known_face_labels = []

    print(f"Loading known faces from '{dataset_path}'...")
    loaded_images = utils.load_images_from_folder(dataset_path)

    if not loaded_images:
        print(f"Warning: No images found in '{dataset_path}'. The system will not recognize anyone.")
        return

    for img_path, image_data in loaded_images:
        # print(f"Processing {img_path}...") # Uncomment for verbose loading
        encoding = utils.encode_face(image_data)
        if encoding is not None:
            known_face_encodings.append(encoding)
            label = utils.extract_label_from_filename(img_path)
            known_face_labels.append(label)
            # print(f"  > Encoded: {label}") # Uncomment for verbose loading
        else:
            print(f"  Warning: No face found or could not encode face in {img_path}.")
    print(f"Successfully loaded {len(known_face_encodings)} known face(s): {list(set(known_face_labels))}")

def main():
    # 1. Load known faces from the dataset
    load_known_faces(DATASET_PATH)

    # 2. Initialize webcam
    video_capture = cv2.VideoCapture(WEBCAM_ID)
    if not video_capture.isOpened():
        print(f"Error: Could not open webcam ID {WEBCAM_ID}. Please check if the webcam is connected and not in use by another application.")
        return

    print("\nStarting webcam...")
    print("Press 'q' on the video window to quit.")

    # Variables for processing optimization
    face_locations = []
    face_encodings_in_frame = []
    face_names = []
    frames_to_skip_counter = 0

    while True:
        # 3. Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to grab frame from webcam. Exiting.")
            break

        # 4. Process frame (detect and recognize faces) only if not skipping
        if frames_to_skip_counter <= 0:
            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            current_face_locations = face_recognition.face_locations(rgb_small_frame)
            current_face_encodings = face_recognition.face_encodings(rgb_small_frame, current_face_locations)

            # Hanya perbarui face_locations dan face_names jika ada perubahan signifikan atau deteksi baru
            # Untuk kesederhanaan, kita akan selalu update jika kita memproses frame ini.
            face_locations = current_face_locations
            face_names = []
            for face_encoding in current_face_encodings:
                name = recognizer.recognize_face(
                    known_face_encodings,
                    known_face_labels,
                    face_encoding,
                    tolerance=RECOGNITION_TOLERANCE
                )
                face_names.append(name)

            # Jika wajah terdeteksi di frame ini, set counter untuk skip beberapa frame berikutnya
            if face_locations:
                frames_to_skip_counter = FRAMES_TO_SKIP_AFTER_DETECTION
            else:
                # Jika tidak ada wajah terdeteksi, jangan skip frame berikutnya
                frames_to_skip_counter = 0
        else:
            frames_to_skip_counter -= 1

        # 5. Display the results
        # Selalu tampilkan bounding box berdasarkan 'face_locations' terakhir yang diketahui
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled
            top = int(top / FRAME_RESIZE_FACTOR)
            right = int(right / FRAME_RESIZE_FACTOR)
            bottom = int(bottom / FRAME_RESIZE_FACTOR)
            left = int(left / FRAME_RESIZE_FACTOR)

            # Tentukan warna box berdasarkan apakah nama dikenali atau tidak
            if name != "Tidak dikenal":
                box_color = (0, 255, 0)  # Hijau untuk wajah yang dikenali
            else:
                box_color = (0, 0, 255)  # Merah untuk wajah "Tidak dikenal"

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) # Teks tetap putih

        # Display the resulting image
        cv2.imshow('Video Face Recognition', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print("Webcam stopped and resources released.")

if __name__ == '__main__':
    main()