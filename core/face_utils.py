import dlib
import numpy as np
from scipy.spatial import distance as dist

# Path ke file model dlib
import os
DLIB_SHAPE_PREDICTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_PATH)

# Indeks landmark untuk mata kiri dan kanan
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

def get_facial_landmarks(gray_image, face_rect):
    """Mendeteksi landmark wajah."""
    shape = predictor(gray_image, face_rect)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye_landmarks):
    """Menghitung Eye Aspect Ratio (EAR)."""
    # Jarak vertikal
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Jarak horizontal
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_face_roi(image, face_rect):
    """Ekstrak Region of Interest (ROI) wajah."""
    (x, y, w, h) = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
    face = image[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200)) # Ukuran standar untuk training
