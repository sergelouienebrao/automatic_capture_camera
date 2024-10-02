import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r'C:/Desktop/mycodes/.venv/Lib/site-packages/haarcascade_frontalface_default.xml')
while True:
     _, frame = cap.read()
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     face = face_cascade.detectMultiScale(gray, 1.3, 5)
     for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 255), 2) 
     cv2.imshow("automatic capture camera", frame)
     if cv2.waitKey(10) == ord("q"): 
      break