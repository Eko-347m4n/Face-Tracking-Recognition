import cv2
import face_recognition
import logging
import time
import threading
import queue
import dlib
from core import recognizer
from core import roi as roi_module
from core.anti_spoofing import BlinkDetector
from core.face_utils import extract_face_roi

logger = logging.getLogger(__name__)

class FaceProcessingPipeline:
    def __init__(self, video_capture, mp_face_detector,
                 known_face_encodings, known_face_labels, known_face_ids,
                 config):
        self.video_capture = video_capture
        self.mp_face_detector = mp_face_detector
        self.known_face_encodings = known_face_encodings
        self.known_face_labels = known_face_labels
        self.known_face_ids = known_face_ids
        self.config = config

        self.stop_event = threading.Event()
        queue_size = self.config.get('QUEUE_SIZE', 2)
        self.frame_q = queue.Queue(maxsize=queue_size)
        self.detection_result_q = queue.Queue(maxsize=queue_size)
        self.display_q = queue.Queue(maxsize=queue_size)

        self.threads = []
        self.active_trackers_with_details = [] # Stores (tracker_obj, (name, id))
        self.skip_counter = 0

        # Adaptive ROI bounding box (x, y, w, h) in original frame coordinates
        self.adaptive_roi = None

        # Initialize BlinkDetector for anti-spoofing
        self.blink_detector = BlinkDetector()

    def _detection_stage(self, original_frame):
        detection_start_time = time.time()
        frame_for_processing = original_frame
        roi_offset_x, roi_offset_y = 0, 0
               # Default ke frame penuh
        actual_roi_coords_on_original_frame = (0, 0, original_frame.shape[1], original_frame.shape[0])

        # Use adaptive ROI if available and enabled
        if self.config.get('ROI_ENABLED', False):
            chosen_roi_source = "None (Full Frame)"
            temp_roi_coords = None

            # Prioritas 1: ROI Adaptif yang Valid
            if self.adaptive_roi: # self.adaptive_roi adalah (x, y, w, h)
                roi_x, roi_y, roi_w, roi_h = self.adaptive_roi
                # Validate ROI dimensions
                if roi_w > 0 and roi_h > 0 and \
                   roi_x >= 0 and roi_y >= 0 and \
                   roi_x + roi_w <= original_frame.shape[1] and \
                   roi_y + roi_h <= original_frame.shape[0]:
                    temp_roi_coords = self.adaptive_roi
                    chosen_roi_source = "Adaptive"
                else:
                    logger.warning(f"Adaptive ROI {self.adaptive_roi} is invalid or out of bounds. Resetting and falling back.")
                    self.adaptive_roi = None # Reset ROI adaptif yang tidak valid
                    # Akan jatuh ke ROI tetap atau logika frame penuh
            # Prioritas 2: ROI Tetap (jika ROI adaptif tidak digunakan atau tidak valid)
            if temp_roi_coords is None and self.config.get('ROI_PARAMS'):
                frame_height, frame_width = original_frame.shape[:2]
                roi_params = self.config['ROI_PARAMS']
                fixed_roi_x, fixed_roi_y, fixed_roi_w, fixed_roi_h = roi_module.define_roi(
                    frame_width, frame_height,
                    roi_x_factor=roi_params.get('x_factor', roi_module.DEFAULT_ROI_X_FACTOR),
                    roi_y_factor=roi_params.get('y_factor', roi_module.DEFAULT_ROI_Y_FACTOR),
                    roi_w_factor=roi_params.get('w_factor', roi_module.DEFAULT_ROI_W_FACTOR),
                    roi_h_factor=roi_params.get('h_factor', roi_module.DEFAULT_ROI_H_FACTOR)
                )
                # define_roi mengembalikan frame penuh jika faktor buruk, jadi periksa apakah itu bukan frame penuh
                if not (fixed_roi_x == 0 and fixed_roi_y == 0 and \
                       fixed_roi_w == frame_width and fixed_roi_h == frame_height):
                    if fixed_roi_w > 0 and fixed_roi_h > 0:
                        temp_roi_coords = (fixed_roi_x, fixed_roi_y, fixed_roi_w, fixed_roi_h)
                        chosen_roi_source = "Fixed"

            # Terapkan ROI yang dipilih jika ada yang diatur dan valid
            if temp_roi_coords: # temp_roi_coords bisa jadi frame penuh dari define_roi
                actual_roi_coords_on_original_frame = temp_roi_coords
                frame_for_processing = roi_module.extract_roi_frame(original_frame, actual_roi_coords_on_original_frame)
                roi_offset_x, roi_offset_y = actual_roi_coords_on_original_frame[0], actual_roi_coords_on_original_frame[1]
                logger.debug(f"Using {chosen_roi_source} ROI: {actual_roi_coords_on_original_frame}")

        else:
            # ROI disabled, process full frame
            actual_roi_coords_on_original_frame = (0, 0, original_frame.shape[1], original_frame.shape[0])
            frame_for_processing = original_frame
            roi_offset_x, roi_offset_y = 0, 0
        
        small_frame_for_processing = cv2.resize(
            frame_for_processing, (0, 0),
            fx=self.config['FRAME_RESIZE_FACTOR'], fy=self.config['FRAME_RESIZE_FACTOR']
        )
        rgb_small_frame = cv2.cvtColor(small_frame_for_processing, cv2.COLOR_BGR2RGB)
        rgb_small_frame.flags.writeable = False

        results = self.mp_face_detector.process(rgb_small_frame)
        rgb_small_frame.flags.writeable = True

        face_locations_in_rgb_small_frame = []
        if results.detections:
            img_h, img_w, _ = rgb_small_frame.shape
            for detection_result in results.detections:
                bbox_mp = detection_result.location_data.relative_bounding_box
                xmin = int(bbox_mp.xmin * img_w)
                ymin = int(bbox_mp.ymin * img_h)
                width = int(bbox_mp.width * img_w)
                height = int(bbox_mp.height * img_h)
                face_locations_in_rgb_small_frame.append((ymin, xmin + width, ymin + height, xmin))

        detection_time_ms = (time.time() - detection_start_time) * 1000
        return (original_frame, rgb_small_frame, face_locations_in_rgb_small_frame,
                actual_roi_coords_on_original_frame, roi_offset_x, roi_offset_y, detection_time_ms)

    def _identification_stage(self, rgb_small_frame, face_locations_in_rgb_small_frame,
                              original_frame_shape, roi_offset_x, roi_offset_y):
        identification_start_time = time.time()
        current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_in_rgb_small_frame)
        recognized_face_details = []
        face_locations_on_original_frame = []
        liveness_statuses = []  # New list to hold liveness status per face

        for i, face_encoding in enumerate(current_face_encodings):
            name, face_id = recognizer.recognize_face(
                self.known_face_encodings, self.known_face_labels, self.known_face_ids,
                face_encoding,
                tolerance=self.config['RECOGNITION_TOLERANCE'],
                default_name=self.config['UNKNOWN_PERSON_LABEL'],
                default_id=self.config['UNKNOWN_FACE_ID']
            )
            recognized_face_details.append((name, face_id))

            # Get corresponding face location in original frame coordinates
            top, right, bottom, left = face_locations_in_rgb_small_frame[i]
            top_rel = int(top / self.config['FRAME_RESIZE_FACTOR'])
            right_rel = int(right / self.config['FRAME_RESIZE_FACTOR'])
            bottom_rel = int(bottom / self.config['FRAME_RESIZE_FACTOR'])
            left_rel = int(left / self.config['FRAME_RESIZE_FACTOR'])

            final_top = top_rel + roi_offset_y
            final_right = right_rel + roi_offset_x
            final_bottom = bottom_rel + roi_offset_y
            final_left = left_rel + roi_offset_x
            face_locations_on_original_frame.append((final_top, final_right, final_bottom, final_left))

            # Prepare dlib.rectangle for landmarks
            face_rect = dlib.rectangle(left=final_left, top=final_top, right=final_right, bottom=final_bottom)

            # Extract grayscale ROI for blink detection
            # Convert rgb_small_frame back to BGR for OpenCV if needed
            # But we have only rgb_small_frame here, so we will extract from original frame in display queue later
            # Instead, we will pass original frame to identification stage in future or refactor

            # For now, we will skip liveness detection here and do it in identification thread with original frame

            # Append placeholder for liveness status
            liveness_statuses.append((False, "Liveness Unknown", 0.0))

        identification_time_ms = (time.time() - identification_start_time) * 1000
        return face_locations_on_original_frame, recognized_face_details, liveness_statuses, identification_time_ms

    def _capture_thread_func(self):
        logger.info("Capture thread started.")
        while not self.stop_event.is_set():
            ret, frame = self.video_capture.read()
            if not ret:
                logger.error("Capture thread: Failed to grab frame. Stopping.")
                self.stop_event.set()
                break
            try:
                self.frame_q.put(frame, timeout=0.1)
            except queue.Full:
                pass # Skip if detection/identification is lagging
            except Exception as e:
                logger.error(f"Capture thread: Error putting frame to queue: {e}")
                self.stop_event.set()
                break
        logger.info("Capture thread finished.")

    def _detection_thread_func(self):
        logger.info("Detection thread started.")
        frames_to_skip_config = self.config.get('FRAMES_TO_SKIP_AFTER_DETECTION', 0)

        while not self.stop_event.is_set():
            try:
                original_frame = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection thread: Error getting frame from queue: {e}")
                self.stop_event.set()
                break

            if self.skip_counter > 0 and self.active_trackers_with_details:
                self.skip_counter -= 1
                
                tracked_locs_orig = []
                tracked_details = []
                current_active_trackers = []
                processing_time_ms = 0 # Minimal for tracking

                track_start_time = time.time()
                for tracker_obj, detail in self.active_trackers_with_details:
                    success, bbox = tracker_obj.update(original_frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        tracked_locs_orig.append((y, x + w, y + h, x)) # top, right, bottom, left
                        tracked_details.append(detail)
                        current_active_trackers.append((tracker_obj, detail))
                    else:
                        logger.debug("Tracker lost an object.")
                self.active_trackers_with_details = current_active_trackers
                processing_time_ms = (time.time() - track_start_time) * 1000
                
                if not self.active_trackers_with_details: # All trackers lost
                    logger.info("All trackers lost during tracking phase. Resetting skip_counter and adaptive_roi.")
                    self.skip_counter = 0
                    self.adaptive_roi = None # Reset adaptive ROI

                # For tracked frames, put directly to display_q
                # Use self.adaptive_roi if available, otherwise full frame, for display purposes.
                roi_for_display_during_tracking = self.adaptive_roi
                if roi_for_display_during_tracking is None:
                    # Fallback if adaptive_roi was reset or never set
                    roi_for_display_during_tracking = (0, 0, original_frame.shape[1], original_frame.shape[0])
                
                # Create placeholder liveness statuses for tracked faces
                liveness_statuses_for_tracked = []
                for _ in tracked_details: # For each tracked face
                    liveness_statuses_for_tracked.append((False, "Tracking", 0.0)) # is_live, status_text, ear_val

                try:
                    # Pass the determined ROI for display
                    self.display_q.put(
                        (original_frame, tracked_locs_orig, tracked_details, liveness_statuses_for_tracked,
                         roi_for_display_during_tracking, processing_time_ms, 0.0),
                        timeout=0.1
                    )
                except queue.Full:
                    pass
                continue # End of processing for skipped/tracked frame

            # Full detection path
            self.active_trackers_with_details = [] # Clear old trackers for a new full detection cycle
            
            (processed_original_frame, rgb_small_frame, face_locs_small,
             roi_coords_orig, r_offset_x, r_offset_y, det_time) = self._detection_stage(original_frame)

            try:
                # Pass data to identification thread
                self.detection_result_q.put(
                    (processed_original_frame, rgb_small_frame, face_locs_small,
                     roi_coords_orig, r_offset_x, r_offset_y, det_time),
                    timeout=0.1
                )
            except queue.Full:
                pass
            except Exception as e:
                logger.error(f"Detection thread: Error putting to detection_result_queue: {e}")
                self.stop_event.set()
                break
        logger.info("Detection thread finished.")

    def _identification_thread_func(self):
        logger.info("Identification thread started.")
        frames_to_skip_config = self.config.get('FRAMES_TO_SKIP_AFTER_DETECTION', 0)

        while not self.stop_event.is_set():
            try:
                (original_frame, rgb_small, face_locs_small, roi_coords,
                 r_offset_x, r_offset_y, det_time) = self.detection_result_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Identification thread: Error getting from detection_result_queue: {e}")
                self.stop_event.set()
                break

            if not face_locs_small: # No faces detected by MediaPipe in detection stage
                self.adaptive_roi = None # Clear adaptive ROI if no faces are detected
                try:
                    self.display_q.put((original_frame, [], [], [], roi_coords, det_time, 0.0), timeout=0.1)
                except queue.Full:
                    pass
                continue

            face_locs_orig, rec_details, liveness_statuses, id_time = self._identification_stage(
                rgb_small, face_locs_small,
                original_frame.shape, r_offset_x, r_offset_y
            )

            # Perform liveness check for each face using BlinkDetector
            updated_liveness_statuses = []
            for i, face_rect_coords in enumerate(face_locs_orig):
                top_o, right_o, bottom_o, left_o = face_rect_coords
                face_id = rec_details[i][1] if i < len(rec_details) else None

                # Create dlib.rectangle for face
                face_rect = dlib.rectangle(left=left_o, top=top_o, right=right_o, bottom=bottom_o)

                # Convert original frame to grayscale for blink detection
                gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

                # Check liveness
                is_live, status_text, ear_val = self.blink_detector.check_liveness(gray_frame, face_rect, face_id)
                updated_liveness_statuses.append((is_live, status_text, ear_val))

            # If faces were identified and skipping is enabled, initialize trackers
            if face_locs_orig and rec_details and frames_to_skip_config > 0:
                # --- Logika Pembaruan ROI Adaptif ---
                # 1. Hitung kotak pembatas ketat dari deteksi saat ini
                current_min_x = min(left_o for (_, _, _, left_o) in face_locs_orig)
                current_min_y = min(top_o for (top_o, _, _, _) in face_locs_orig)
                current_max_x = max(right_o for (_, right_o, _, _) in face_locs_orig)
                current_max_y = max(bottom_o for (_, _, bottom_o, _) in face_locs_orig)

                # 2. Tambahkan padding ke kotak pembatas ini
                padding = self.config.get('ADAPTIVE_ROI_PADDING', 30) # Padding default, dapat dikonfigurasi
                
                # Koordinat dengan padding untuk set deteksi saat ini
                padded_current_x1 = max(0, current_min_x - padding)
                padded_current_y1 = max(0, current_min_y - padding)
                padded_current_x2 = min(original_frame.shape[1], current_max_x + padding)
                padded_current_y2 = min(original_frame.shape[0], current_max_y + padding)

                new_roi_x, new_roi_y, new_roi_w, new_roi_h = 0,0,0,0

                # 3. Jika self.adaptive_roi None, inisialisasi dengan deteksi saat ini yang diberi padding.
                if self.adaptive_roi is None:
                    new_roi_x = padded_current_x1
                    new_roi_y = padded_current_y1
                    new_roi_w = padded_current_x2 - padded_current_x1
                    new_roi_h = padded_current_y2 - padded_current_y1
                else:
                    # 4. Jika self.adaptive_roi ada, hitung gabungan (union) dengan deteksi saat ini yang diberi padding.
                    # Ini memastikan ROI tidak menyusut dan hanya meluas jika perlu.
                    old_roi_x, old_roi_y, old_roi_w, old_roi_h = self.adaptive_roi
                    old_roi_x1, old_roi_y1 = old_roi_x, old_roi_y
                    old_roi_x2, old_roi_y2 = old_roi_x + old_roi_w, old_roi_y + old_roi_h

                    # Koordinat gabungan
                    union_x1 = min(old_roi_x1, padded_current_x1)
                    union_y1 = min(old_roi_y1, padded_current_y1)
                    union_x2 = max(old_roi_x2, padded_current_x2)
                    union_y2 = max(old_roi_y2, padded_current_y2)

                    new_roi_x = union_x1
                    new_roi_y = union_y1
                    new_roi_w = union_x2 - union_x1
                    new_roi_h = union_y2 - union_y1
                
                # 5. Perbarui self.adaptive_roi jika ROI baru valid
                if new_roi_w > 0 and new_roi_h > 0:
                    # Update adaptive_roi only if new ROI is larger or equal in size (prevent shrinking)
                    if self.adaptive_roi is None:
                        self.adaptive_roi = (int(new_roi_x), int(new_roi_y), int(new_roi_w), int(new_roi_h))
                        logger.debug(f"Adaptive ROI initialized to: {self.adaptive_roi}")
                    else:
                        old_x, old_y, old_w, old_h = self.adaptive_roi
                        # Calculate areas
                        old_area = old_w * old_h
                        new_area = int(new_roi_w) * int(new_roi_h)
                        if new_area >= old_area:
                            self.adaptive_roi = (int(new_roi_x), int(new_roi_y), int(new_roi_w), int(new_roi_h))
                            logger.debug(f"Adaptive ROI updated to larger area: {self.adaptive_roi}")
                        else:
                            logger.debug(f"Adaptive ROI not updated as new union area is smaller. Kept old ROI: {self.adaptive_roi}. Proposed smaller union: {(int(new_roi_x), int(new_roi_y), int(new_roi_w), int(new_roi_h))}")
                else:
                    logger.warning(f"Adaptive ROI calculation resulted in non-positive W/H ({new_roi_w}, {new_roi_h}). Resetting adaptive ROI.")
                    self.adaptive_roi = None 

                # --- Inisialisasi Tracker (seperti sebelumnya) ---
                temp_trackers = []
                # Calculate adaptive ROI bounding box around all detected faces with margin
                margin = 20  # pixels margin around faces for ROI
                min_x = min(left for (_, _, _, left) in face_locs_orig)
                min_y = min(top for (top, _, _, _) in face_locs_orig)

                for i, (top_o, right_o, bottom_o, left_o) in enumerate(face_locs_orig):
                    detail = rec_details[i]
                    x_o, y_o = left_o, top_o
                    w_o, h_o = right_o - left_o, bottom_o - top_o

                    if w_o > 0 and h_o > 0:
                        tracker = cv2.TrackerCSRT_create()
                        try:
                            # Initialize tracker on the original frame with original coordinates
                            tracker.init(original_frame, (x_o, y_o, w_o, h_o))
                            temp_trackers.append((tracker, detail))
                        except Exception as e_tracker:
                            logger.warning(f"Failed to initialize tracker for a face: {e_tracker}")
                
                if temp_trackers:
                    self.active_trackers_with_details = temp_trackers
                    self.skip_counter = frames_to_skip_config # Set skip counter for detection thread
            else:
                # Ensure trackers are cleared if no faces or skipping disabled
                self.active_trackers_with_details = []
                self.skip_counter = 0
                self.adaptive_roi = None  # Clear adaptive ROI if no faces

            try:
                self.display_q.put((original_frame, face_locs_orig, rec_details, liveness_statuses, roi_coords, det_time, id_time), timeout=0.1)
            except queue.Full:
                pass
            except Exception as e:
                logger.error(f"Identification thread: Error putting to display_queue: {e}")
                self.stop_event.set()
                break
        logger.info("Identification thread finished.")

    def start(self):
        logger.info("Starting processing pipeline threads...")
        capture_t = threading.Thread(target=self._capture_thread_func, name="CaptureThread")
        self.threads.append(capture_t)

        detection_t = threading.Thread(target=self._detection_thread_func, name="DetectionThread")
        self.threads.append(detection_t)

        identification_t = threading.Thread(target=self._identification_thread_func, name="IdentificationThread")
        self.threads.append(identification_t)

        for t in self.threads:
            t.start()
        logger.info("All pipeline threads started.")

    def stop(self):
        logger.info("Stopping processing pipeline threads...")
        self.stop_event.set()
        for t in self.threads:
            try:
                t.join(timeout=2.0)
                if t.is_alive():
                    logger.warning(f"Thread {t.name} did not terminate in time.")
            except Exception as e:
                logger.error(f"Error joining thread {t.name}: {e}")
        logger.info("Processing pipeline threads stopped.")

    def get_display_data(self, timeout=0.1):
        """Gets data from the display queue."""
        try:
            return self.display_q.get(timeout=timeout)
        except queue.Empty:
            return None