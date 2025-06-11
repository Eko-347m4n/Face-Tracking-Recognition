import time
from collections import defaultdict
from core.face_utils import eye_aspect_ratio, get_facial_landmarks, L_START, L_END, R_START, R_END
import dlib
import os

class BlinkDetector:
    """
    Manages blink detection state for multiple faces.
    Uses EAR (Eye Aspect Ratio) to detect blinks and determine liveness.
    """

    EAR_THRESHOLD = 0.23  # Threshold for EAR to consider eye closed
    EAR_CONSEC_FRAMES_BLINK = 2  # Number of consecutive frames EAR below threshold to count as a blink
    MIN_BLINKS_FOR_LIVE = 1  # Minimum blinks in observation period to consider live
    BLINK_CHECK_DURATION = 3  # Seconds to observe blinks
    FRAMES_TO_RESET_BLINK_COUNT = 30  # Frames without face to reset blink count

    def __init__(self):
        # State per face_id
        self.blink_counters = defaultdict(int)
        self.ear_low_frames_counts = defaultdict(int)
        self.frames_since_blink_check_start = defaultdict(int)
        self.blink_check_active_for_face_id = None
        self.last_blink_time = defaultdict(lambda: time.time())
        self.frames_no_face_detected = defaultdict(int)

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(model_path)
        self.detector = dlib.get_frontal_face_detector()

    def check_liveness(self, gray_frame, face_rect, face_id):
        """
        Check liveness for a given face using blink detection.
        :param gray_frame: Grayscale image frame
        :param face_rect: dlib.rectangle for face location
        :param face_id: unique identifier for the face
        :return: (is_live: bool, status_text: str, ear: float)
        """
        shape = self.predictor(gray_frame, face_rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = coords[L_START:L_END]
        right_eye = coords[R_START:R_END]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Reset state if face_id changed
        if self.blink_check_active_for_face_id != face_id:
            self.blink_check_active_for_face_id = face_id
            self.blink_counters[face_id] = 0
            self.frames_since_blink_check_start[face_id] = 0
            self.ear_low_frames_counts[face_id] = 0
            self.last_blink_time[face_id] = time.time()

        self.frames_since_blink_check_start[face_id] += 1

        is_live = False
        status_text = f"EAR: {ear:.2f}"

        if ear < self.EAR_THRESHOLD:
            self.ear_low_frames_counts[face_id] += 1
        else:
            if self.ear_low_frames_counts[face_id] >= self.EAR_CONSEC_FRAMES_BLINK:
                self.blink_counters[face_id] += 1
                self.last_blink_time[face_id] = time.time()
            self.ear_low_frames_counts[face_id] = 0

        if self.blink_counters[face_id] >= self.MIN_BLINKS_FOR_LIVE:
            is_live = True
            status_text += " | Live (Blinks OK)"
        elif (time.time() - self.last_blink_time[face_id]) > self.BLINK_CHECK_DURATION:
            is_live = False
            status_text += " | Spoof? (No blinks)"
            self.blink_counters[face_id] = 0
            self.frames_since_blink_check_start[face_id] = 0
            self.last_blink_time[face_id] = time.time()
        else:
            status_text += f" | Checking... Blinks: {self.blink_counters[face_id]}"

        if self.frames_since_blink_check_start[face_id] > (30 * self.BLINK_CHECK_DURATION) and not is_live:
            if self.blink_counters[face_id] < self.MIN_BLINKS_FOR_LIVE:
                is_live = False
                status_text = f"EAR: {ear:.2f} | Spoof (Timeout, Blinks: {self.blink_counters[face_id]})"
                self.blink_counters[face_id] = 0
                self.frames_since_blink_check_start[face_id] = 0
                self.last_blink_time[face_id] = time.time()

        return is_live, status_text, ear
