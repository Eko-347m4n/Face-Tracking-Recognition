import face_recognition
import numpy as np

def recognize_face(known_face_encodings, known_face_labels, face_encoding_to_check, tolerance=0.6):
    if not known_face_encodings:
        return "Tidak dikenal" 

    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=tolerance)
    name = "Tidak dikenal"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_labels[best_match_index]

    return name
