import face_recognition
import os
import logging
from pathlib import Path # Import pathlib

logger = logging.getLogger(__name__)

def load_images_from_folder(folder_path_str): # Renaming input arg for clarity
    """
    Loads images recursively from subfolders in the given folder_path.
    - For images in subdirectories: label is the subdirectory name.
    - For images directly in folder_path: label is derived from the filename.
    Returns a list of tuples: (image_path, image_data, label, modification_time).
    """
    images_data = []
    base_path = Path(folder_path_str) # Use pathlib.Path

    if not base_path.is_dir():
        logger.error(f"Dataset folder '{base_path}' not found.")
        return images_data

    logger.info(f"Scanning for images in '{base_path}' and its subdirectories...")

    # Using rglob to find all relevant files recursively
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            label = ""
            # Determine label based on file location
            if file_path.parent == base_path:
                # File is in the root of folder_path, derive label from filename
                label = extract_label_from_filename(file_path.name)
            else:
                # File is in a subdirectory, use subdirectory name as label
                label = file_path.parent.name
            
            try:
                image_obj = face_recognition.load_image_file(str(file_path)) # face_recognition needs string path
                mod_time = file_path.stat().st_mtime
                images_data.append((str(file_path), image_obj, label, mod_time))
                logger.debug(f"Loaded image '{file_path}' with label '{label}' (mod_time: {mod_time})")
            except Exception as e:
                logger.error(f"Error loading image {file_path}: {e}")
    return images_data

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
