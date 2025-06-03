import os
import pickle
import logging
from core import utils

logger = logging.getLogger(__name__)

def load_known_faces_from_dataset(dataset_path):
    """
    Loads known faces from images in the dataset_path, using a pickle file for caching.
    Returns (known_face_encodings, known_face_labels, known_face_ids).
    """
    known_face_encodings = []
    known_face_labels = []
    known_face_ids = []

    pickle_file_name = "dataset_encodings.pkl"
    pickle_file_path = os.path.join(dataset_path, pickle_file_name)

    # 1. Attempt to load from pickle file
    if os.path.exists(pickle_file_path):
        logger.info(f"Attempting to load known faces from '{pickle_file_path}'...")
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
                    logger.info(f"Successfully loaded {len(known_face_encodings)} known face(s) from pickle: {list(unique_labels_with_ids)}")
                    return known_face_encodings, known_face_labels, known_face_ids
                else:
                    logger.warning(f"Pickle file '{pickle_file_path}' is empty or malformed. Will re-encode dataset.")
        except Exception as e:
            logger.warning(f"Could not load from pickle file '{pickle_file_path}' (error: {e}). Will re-encode dataset.")
            try:
                os.remove(pickle_file_path)
                logger.info(f"Removed potentially corrupted pickle file: '{pickle_file_path}'.")
            except OSError as oe:
                logger.warning(f"Could not remove pickle file '{pickle_file_path}': {oe}")

    # 2. If pickle loading failed or file didn't exist, load from images and generate encodings
    logger.info(f"Loading known faces from image dataset '{dataset_path}' and generating new encodings...")
    loaded_images = utils.load_images_from_folder(dataset_path)

    if not loaded_images:
        logger.warning(f"No images found in '{dataset_path}'. The system will not recognize anyone.")
        if os.path.exists(pickle_file_path):
            try:
                logger.info(f"No images found for re-encoding. Removing existing pickle file '{pickle_file_path}'.")
                os.remove(pickle_file_path)
            except OSError as e:
                logger.warning(f"Could not remove existing pickle file '{pickle_file_path}': {e}")
        return known_face_encodings, known_face_labels, known_face_ids # Return empty lists

    id_counter = 0
    for img_path, image_data in loaded_images:
        encoding = utils.encode_face(image_data)
        if encoding is not None:
            known_face_encodings.append(encoding)
            label = utils.extract_label_from_filename(img_path)
            known_face_labels.append(label)
            known_face_ids.append(f"fid_{id_counter}")
            id_counter += 1
        else:
            logger.warning(f"No face found or could not encode face in {img_path}.")

    if known_face_encodings:
        unique_labels_with_ids = {f"{label} (ID: {id_val})" for label, id_val in zip(known_face_labels, known_face_ids)}
        logger.info(f"Successfully encoded {len(known_face_encodings)} known face(s) from images: {list(unique_labels_with_ids)}")
        try:
            os.makedirs(dataset_path, exist_ok=True)
            with open(pickle_file_path, 'wb') as f:
                pickle.dump({'encodings': known_face_encodings, 'labels': known_face_labels, 'ids': known_face_ids}, f)
            logger.info(f"Encodings saved successfully to '{pickle_file_path}'.")
        except Exception as e:
            logger.error(f"Failed to save encodings to pickle file '{pickle_file_path}': {e}")
    else:
        logger.warning("No faces were encoded from the dataset. No new pickle file created/updated.")
        if os.path.exists(pickle_file_path):
            try:
                logger.info(f"No new encodings generated. Removing existing pickle file '{pickle_file_path}'.")
                os.remove(pickle_file_path)
            except OSError as e:
                logger.warning(f"Could not remove existing pickle file '{pickle_file_path}': {e}")
                
    return known_face_encodings, known_face_labels, known_face_ids