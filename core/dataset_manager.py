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

    def current_images_mod_times(images_with_labels):
        # Create a dict mapping image path to modification time
        return {img_path: mod_time for img_path, _, _, mod_time in images_with_labels}

    # 1. Attempt to load from pickle file
    if os.path.exists(pickle_file_path):
        logger.info(f"Attempting to load known faces from '{pickle_file_path}'...")
        try:
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
                if 'encodings' in data and 'labels' in data and 'ids' in data and 'mod_times' in data and \
                   data['encodings'] and data['labels'] and data['ids'] and data['mod_times'] and \
                   len(data['encodings']) == len(data['labels']) == len(data['ids']):
                    # Load current images to compare modification times
                    loaded_images_with_labels = utils.load_images_from_folder(dataset_path)
                    current_mod_times = current_images_mod_times(loaded_images_with_labels)
                    # Compare stored mod_times with current mod_times
                    if data['mod_times'] == current_mod_times:
                        known_face_encodings = data['encodings']
                        known_face_labels = data['labels']
                        known_face_ids = data['ids']
                        unique_labels_with_ids = {f"{label} (ID: {id_val})" for label, id_val in zip(known_face_labels, known_face_ids)}
                        logger.info(f"Successfully loaded {len(known_face_encodings)} known face(s) from pickle: {list(unique_labels_with_ids)}")
                        return known_face_encodings, known_face_labels, known_face_ids
                    else:
                        logger.info("Dataset has changed since last encoding. Re-encoding required.")
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
    loaded_images_with_labels = utils.load_images_from_folder(dataset_path)

    if not loaded_images_with_labels:
        logger.warning(f"No images found in subdirectories of '{dataset_path}'. The system will not recognize anyone.")
        if os.path.exists(pickle_file_path):
            try:
                logger.info(f"No images found in subdirectories for re-encoding. Removing existing pickle file '{pickle_file_path}'.")
                os.remove(pickle_file_path)
            except OSError as e:
                logger.warning(f"Could not remove existing pickle file '{pickle_file_path}': {e}")
        return known_face_encodings, known_face_labels, known_face_ids # Return empty lists

    id_counter = 0
    for img_path, image_data, label_from_dir, mod_time in loaded_images_with_labels:
        encoding = utils.encode_face(image_data)
        if encoding is not None:
            known_face_encodings.append(encoding)
            # Use the label derived from the directory structure
            known_face_labels.append(label_from_dir)
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
                pickle.dump({
                    'encodings': known_face_encodings,
                    'labels': known_face_labels,
                    'ids': known_face_ids,
                    'mod_times': current_images_mod_times(loaded_images_with_labels)
                }, f)
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
