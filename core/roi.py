import cv2
import logging
import os # Ditambahkan untuk pengujian standalone

# --- Konfigurasi Default untuk ROI ---
DEFAULT_ROI_X_FACTOR = 0.25  # Mulai dari 25% lebar frame
DEFAULT_ROI_Y_FACTOR = 0.1   # Mulai dari 10% tinggi frame
DEFAULT_ROI_W_FACTOR = 0.5   # Lebar ROI 50% dari lebar frame
DEFAULT_ROI_H_FACTOR = 0.7   # Tinggi ROI 70% dari tinggi frame

logger = logging.getLogger(__name__)

def define_roi(frame_width, frame_height,
               roi_x_factor=DEFAULT_ROI_X_FACTOR,
               roi_y_factor=DEFAULT_ROI_Y_FACTOR,
               roi_w_factor=DEFAULT_ROI_W_FACTOR,
               roi_h_factor=DEFAULT_ROI_H_FACTOR):
    """
    Mendefinisikan koordinat Region of Interest (ROI).

    Args:
        frame_width (int): Lebar frame penuh.
        frame_height (int): Tinggi frame penuh.
        roi_x_factor (float): Faktor untuk menentukan koordinat x awal ROI.
        roi_y_factor (float): Faktor untuk menentukan koordinat y awal ROI.
        roi_w_factor (float): Faktor untuk menentukan lebar ROI.
        roi_h_factor (float): Faktor untuk menentukan tinggi ROI.

    Returns:
        tuple: (roi_x, roi_y, roi_w, roi_h) koordinat ROI.
               Mengembalikan (0, 0, frame_width, frame_height) jika faktor tidak valid.
    """
    if not (0 <= roi_x_factor < 1 and \
            0 <= roi_y_factor < 1 and \
            0 < roi_w_factor <= 1 and \
            0 < roi_h_factor <= 1 and \
            roi_x_factor + roi_w_factor <= 1 and \
            roi_y_factor + roi_h_factor <= 1):
        logger.warning(f"Faktor ROI tidak valid. Menggunakan frame penuh. "
                       f"x_f:{roi_x_factor}, y_f:{roi_y_factor}, w_f:{roi_w_factor}, h_f:{roi_h_factor}")
        return 0, 0, frame_width, frame_height

    roi_x = int(frame_width * roi_x_factor)
    roi_y = int(frame_height * roi_y_factor)
    roi_w = int(frame_width * roi_w_factor)
    roi_h = int(frame_height * roi_h_factor)

    # Pastikan ROI berada dalam batas frame
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_w = min(roi_w, frame_width - roi_x)
    roi_h = min(roi_h, frame_height - roi_y)

    return roi_x, roi_y, roi_w, roi_h

def extract_roi_frame(frame, roi_coords):
    """
    Mengekstrak ROI dari frame yang diberikan.

    Args:
        frame (numpy.ndarray): Frame gambar penuh.
        roi_coords (tuple): (roi_x, roi_y, roi_w, roi_h) koordinat.

    Returns:
        numpy.ndarray: Gambar ROI yang dipotong.
                       Mengembalikan frame asli jika roi_coords tidak valid untuk pemotongan.
    """
    roi_x, roi_y, roi_w, roi_h = roi_coords
    if roi_w <= 0 or roi_h <= 0:
        logger.warning(f"Dimensi ROI tidak valid untuk pemotongan: w={roi_w}, h={roi_h}. Mengembalikan frame asli.")
        return frame
    return frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

def draw_roi_boundary(frame, roi_coords, color=(255, 0, 0), thickness=2):
    """
    Menggambar batas ROI pada frame yang diberikan.

    Args:
        frame (numpy.ndarray): Frame untuk digambar (akan dimodifikasi).
        roi_coords (tuple): (roi_x, roi_y, roi_w, roi_h) koordinat.
        color (tuple): Warna BGR untuk persegi panjang.
        thickness (int): Ketebalan garis persegi panjang.
    """
    roi_x, roi_y, roi_w, roi_h = roi_coords
    if roi_w > 0 and roi_h > 0: # Hanya gambar jika ROI valid
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, thickness)
    
        cv2.putText(frame, 'AREA SCAN', (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

if __name__ == '__main__':
    # Contoh penggunaan untuk menguji roi.py secara independen
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Menguji modul ROI secara independen...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Tidak bisa membuka kamera.")
        exit()

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        logger.error(f"File Haar Cascade tidak ditemukan di {face_cascade_path}. Deteksi wajah dalam tes tidak akan berfungsi.")
        face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Tidak bisa membaca frame.")
            break

        frame_height, frame_width = frame.shape[:2]
        roi_x, roi_y, roi_w, roi_h = define_roi(frame_width, frame_height) # Menggunakan faktor default
        roi_coords = (roi_x, roi_y, roi_w, roi_h)
        roi_frame = extract_roi_frame(frame, roi_coords)

        if face_cascade and roi_frame.size > 0:
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            faces_in_roi = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces_in_roi:
                cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + roi_x + w, y + roi_y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Wajah', (x + roi_x, y + roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        draw_roi_boundary(frame, roi_coords)
        cv2.imshow('ROI Test Frame', frame)
        if roi_frame.size > 0:
            cv2.imshow('Extracted ROI', roi_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Tes modul ROI selesai.")
