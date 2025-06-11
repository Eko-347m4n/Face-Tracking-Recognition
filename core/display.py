import cv2

def draw_detections_on_frame(frame, face_locations, face_details, liveness_statuses, frame_resize_factor, unknown_person_label, fps=0.0, response_time_ms=0.0):
    # Display FPS and Response Time (Waktu Pemrosesan)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (0, 0, 0)  # Hitam
    bg_color = (255, 255, 255) # Putih
    thickness = 1
    line_type = cv2.LINE_AA
    
    text_start_x = 10
    margin = 5 
    current_y_baseline = 30 # Baseline Y awal untuk teks pertama

    # Tampilan FPS
    fps_text = f"FPS: {fps:.2f}"
    (w_fps, h_fps), baseline_fps_val = cv2.getTextSize(fps_text, font, font_scale, thickness)
    
    cv2.rectangle(frame, (text_start_x - margin, current_y_baseline - h_fps - margin), \
                         (text_start_x + w_fps + margin, current_y_baseline + baseline_fps_val + margin), bg_color, cv2.FILLED)
    cv2.putText(frame, fps_text, (text_start_x, current_y_baseline), font, font_scale, text_color, thickness, line_type)
    
    current_y_baseline += (h_fps + baseline_fps_val + margin * 2) # Update Y untuk teks berikutnya

    # Tampilan Waktu Pemrosesan (Response Time)
    response_text = f"Proc. Time: {response_time_ms:.2f} ms"
    (w_resp, h_resp), baseline_resp_val = cv2.getTextSize(response_text, font, font_scale, thickness)

    cv2.rectangle(frame, (text_start_x - margin, current_y_baseline - h_resp - margin), \
                         (text_start_x + w_resp + margin, current_y_baseline + baseline_resp_val + margin), bg_color, cv2.FILLED)
    cv2.putText(frame, response_text, (text_start_x, current_y_baseline), font, font_scale, text_color, thickness, line_type)

    # --- Gambar Deteksi Wajah (hanya jika face_locations tidak kosong) ---
    if face_locations: # Periksa apakah ada wajah untuk digambar
        for (top, right, bottom, left), (name, face_id), (is_live, status_text, ear_val) in zip(face_locations, face_details, liveness_statuses):
        # Scale back up face locations to original frame size
            top_orig = int(top / frame_resize_factor)
            right_orig = int(right / frame_resize_factor)
            bottom_orig = int(bottom / frame_resize_factor)
            left_orig = int(left / frame_resize_factor)

            box_color = (0, 255, 0) if name != unknown_person_label else (0, 0, 255)
            # Change box color to red if spoof detected (not live)
            if not is_live:
                box_color = (0, 0, 255)

            cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), box_color, 2)

            # Siapkan teks untuk ditampilkan
            display_text = f"{name} (ID: {face_id})" if name != unknown_person_label else name
            display_text += f" | {status_text}"
            
            face_label_font = cv2.FONT_HERSHEY_DUPLEX
            face_label_font_scale = 0.5 
            face_label_thickness = 1
            (text_width, text_height), baseline_label = cv2.getTextSize(display_text, face_label_font, face_label_font_scale, face_label_thickness)
            
            # Gambar persegi panjang terisi untuk latar belakang teks di bawah kotak wajah
            rect_top = bottom_orig - text_height - baseline_label - 5 # Sesuaikan posisi agar pas
            rect_bottom = bottom_orig
            
            cv2.rectangle(frame, (left_orig, rect_top), (left_orig + text_width + 10, rect_bottom), box_color, cv2.FILLED)
            cv2.putText(frame, display_text, (left_orig + 6, bottom_orig - baseline_label - 2), face_label_font, face_label_font_scale, (255, 255, 255), face_label_thickness)


def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False

def destroy_all_windows():
    cv2.destroyAllWindows()
