# modification start
# countdown
# flipped cam
# multiface detection
#beep sound

import cv2
import datetime
import time
import winsound
import numpy as np

cap = cv2.VideoCapture(0)

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(r'C:/Desktop/mycodes/.venv/Lib/site-packages/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(r'C:/Downloads/haarcascade_smile.xml')

if face_cascade.empty() or smile_cascade.empty():
    print("Error loading cascade files.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

def apply_filter(frame, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'sepia':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_filter).clip(0, 255).astype(np.uint8)
    elif filter_type == 'cartoon':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)
    return frame

capture_delay = 3
capture_interval = 5  # Interval in seconds between captures
countdown_start_time = None
countdown_in_progress = False
last_capture_time = 0  # Tracks the time of the last capture
selected_filter = 'none'  # Default filter

while True: 
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    original_frame = frame.copy()  # Keep an unaltered copy for saving

    # Apply the selected filter
    if selected_filter != 'none':
        frame = apply_filter(frame, selected_filter)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if selected_filter != 'grayscale' else frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
        face_roi = frame[y: y + height, x: x + width]
        gray_roi = gray[y: y + height, x: x + width]
        smiles = smile_cascade.detectMultiScale(
            gray_roi, scaleFactor=1.7, minNeighbors=20, minSize=(15, 15)
        )

        # Start countdown only if the interval since the last capture has passed
        if len(smiles) > 0 and not countdown_in_progress and (time.time() - last_capture_time) >= capture_interval:
            countdown_start_time = time.time()  # Start countdown
            countdown_in_progress = True

    # Countdown and capture logic
    if countdown_in_progress and countdown_start_time is not None:
        elapsed_time = time.time() - countdown_start_time
        countdown_time = capture_delay - int(elapsed_time)

        if countdown_time > 0:
            cv2.putText(frame, f"Capturing in {countdown_time}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # Play sound during countdown
            if elapsed_time - int(elapsed_time) < 0.1:  # Trigger beep once per second
                winsound.Beep(1000, 100)  # Beep for 100ms
        elif countdown_time == 0:
            # Play a sound when countdown hits 0
            winsound.Beep(1200, 200)  # Beep for 200ms
            # Capture photo (save the unaltered frame)
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"selfie-{time_stamp}.png"
            if selected_filter != 'none':
                filtered_frame = apply_filter(original_frame, selected_filter)
                cv2.imwrite(file_name, filtered_frame)
            else:
                cv2.imwrite(file_name, original_frame)
            countdown_in_progress = False
            countdown_start_time = None
            last_capture_time = time.time()  # Update last capture time
            winsound.Beep(1500, 500)  # Play sound after capture

    cv2.putText(frame, "Press 'f' to toggle filters", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Current filter: {selected_filter}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("automatic capture camera", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Toggle filters
        filters = ['none', 'grayscale', 'sepia', 'cartoon']
        current_index = filters.index(selected_filter)
        selected_filter = filters[(current_index + 1) % len(filters)]

cap.release()
cv2.destroyAllWindows()