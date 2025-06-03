import cv2
import logging

logger = logging.getLogger(__name__)

def initialize_webcam(webcam_id: int):
    """Initializes and returns the webcam capture object."""
    video_capture = cv2.VideoCapture(webcam_id)
    if not video_capture.isOpened():
        logger.error(f"Could not open webcam ID {webcam_id}. Please check connectivity.")
        return None
    return video_capture