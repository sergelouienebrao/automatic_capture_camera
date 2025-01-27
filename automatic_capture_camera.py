# modification start
# countdown
# flipped cam
# multiface detection


import cv2
import datetime
import time

cap = cv2.VideoCapture(0)

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(r'C:/Desktop/mycodes/.venv/Lib/site-packages/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(r'C:/Downloads/haarcascade_smile.xml')

if face_cascade.empty() or smile_cascade.empty():
    print("Error loading cascade files.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

capture_delay = 3
countdown_start_time = None
countdown_in_progress = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
        face_roi = frame[y: y + height, x: x + width]
        gray_roi = gray[y: y + height, x: x + width]
        smiles = smile_cascade.detectMultiScale(
            gray_roi, scaleFactor=1.7, minNeighbors=20, minSize=(15, 15)
        )

        if len(smiles) > 0 and not countdown_in_progress:
            countdown_start_time = time.time()  # Start countdown
            countdown_in_progress = True

    # Countdown and capture logic
    if countdown_in_progress and countdown_start_time is not None:
        elapsed_time = time.time() - countdown_start_time
        countdown_time = capture_delay - int(elapsed_time)

        if countdown_time > 0:
            cv2.putText(frame, f"Capturing in {countdown_time}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Capture photo
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"selfie-{time_stamp}.png"
            cv2.imwrite(file_name, frame)
            countdown_in_progress = False
            countdown_start_time = None

    cv2.imshow("automatic capture camera", frame)

    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
