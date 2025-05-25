import face_recognition
import os

def load_images_from_folder(folder_path):
    images = []
    if not os.path.isdir(folder_path):
        print(f"Error: Dataset folder '{folder_path}' not found.")
        return images

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(img_path) 
            images.append((img_path, image))
    return images

def encode_face(image_data):
    face_encodings = face_recognition.face_encodings(image_data)
    if face_encodings:
        return face_encodings[0] 
    return None

def extract_label_from_filename(filename):
    base_name = os.path.basename(filename)
    # Get the part before the first underscore, or the whole name if no underscore
    name_candidate = base_name.split('_')[0]
    # Remove the extension from this candidate
    label = os.path.splitext(name_candidate)[0]
    return label
