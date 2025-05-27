import face_recognition
import numpy as np

def recognize_face(known_face_encodings, known_face_labels, known_face_ids, face_encoding_to_check, tolerance=0.6, default_name="Tidak dikenal", default_id="---"):
    """
    Compares a face encoding to a list of known face encodings and returns the name and ID of the best match.
    """
    if not known_face_encodings: # Tambahkan pemeriksaan jika list kosong
        return default_name, default_id

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance)
    name = default_name
    face_id = default_id

    # Calculate distances to all known faces
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)

    # Find the best match (smallest distance) if any match was found
    if True in matches:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]: # Verifikasi bahwa indeks terbaik memang cocok
            name = known_face_labels[best_match_index]
            face_id = known_face_ids[best_match_index]
            
    return name, face_id
