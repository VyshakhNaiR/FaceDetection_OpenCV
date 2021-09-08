import cv2
import numpy as np
facecas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyecas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
url = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(url)
while True:
    ret, feed = cap.read()
    gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
    faces = facecas.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(feed, (x, y), (x + w, y + h), (255, 0, 0), 2)
        r_gray = gray[y:y+h, x:x+w]
        r_color = feed[y:y + h, x:x + w]
        eye = eyecas.detectMultiScale(r_gray)
        for (Ex, Ey, Ew, Eh) in eye:
            cv2.rectangle(r_color, (Ex, Ey), (Ex + Ew, Ey + Eh), (0, 255, 0), 2)
    cv2.imshow("live", feed)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
