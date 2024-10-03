import cv2
import datetime
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r'C:/Desktop/mycodes/.venv/Lib/site-packages/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(r'C:/Downloads/haarcascade_smile.xml')
while True:
     _, frame = cap.read()
     original_frame = frame.copy()
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     face = face_cascade.detectMultiScale(gray, 1.3, 5)
     for x, y, width, height in face:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 225, 255), 2) 
        face_roi = frame[y: y + height, x: x + width]
        gray_roi = gray[y: y + height, x: x + width]
        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)
        for x1, y1, width1, height1 in smile:
            cv2.rectangle(face_roi, (x1, y1), (x1 + width1, y1 + height1), (0, 0, 255), 2)
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"selfie-{time_stamp}.png"
            cv2.imwrite(file_name, original_frame) 
     cv2.imshow("automatic capture camera", frame)
     if cv2.waitKey(10) == ord("q"): 
      break 