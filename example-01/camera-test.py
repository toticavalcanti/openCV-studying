import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

url = 'http://192.168.0.246:8080/video'
cap = cv2.VideoCapture(url)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        color = (255, 0, 0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)


    cv2.imshow('frame',frame)
    if  cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # if frame is not None:
    #     cv2.imshow('frame',frame)
    # q = cv2.waitKey(1)
    # if q == ord("q"):
    #     break
cap.release()
cv2.destroyAllWindows()