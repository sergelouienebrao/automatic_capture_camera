#modification start
import cv2
import datetime
import time

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(r'C:/Desktop/mycodes/.venv/Lib/site-packages/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(r'C:/Downloads/haarcascade_smile.xml')

last_capture_time = 0
capture_delay = 3

while True:
     _, frame = cap.read()
     original_frame = frame.copy()
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5, minSize = (100, 100))

     for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 225, 225), 2) 
        face_roi = frame[y: y + height, x: x + width]
        gray_roi = gray[y: y + height, x: x + width]
        smiles = smile_cascade.detectMultiScale(gray_roi, scaleFactor = 1.8, minNeighbors = 25, minSize= (25, 25))

        for x1, y1, width1, height1 in smiles:
            cv2.rectangle(face_roi, (x1, y1), (x1 + width1, y1 + height1), (0, 0, 255), 2)
            current_time = time.time()
            if current_time - last_capture_time >= capture_delay:
               for i in range(3, 0, -1):
                  countdown_frame = frame.copy()
                  cv2.putText(countdown_frame, f"Capturing in {i}", (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 2)
                  cv2.imshow("automatic capture camera", countdown_frame)
                  cv2.waitKey(1000)

            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"selfie-{time_stamp}.png"
            cv2.imwrite(file_name, original_frame) 
    
     cv2.imshow("automatic capture camera", frame)
     if cv2.waitKey(10) == ord("q"): 
      break 
     
# cap.release()
# cv2.destroyAllWindows()
