#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:35:55 2020

@author: rahul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:17:03 2020

@author: rahul
"""

import cv2
import skvideo.io

person_classifier= cv2.CascadeClassifier('haarcascade_fullbody.xml')
eye_classifier= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect(gray, frame):
    people= person_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in people:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
        print(x,y,w,h)
    return frame




video_capture= cv2.VideoCapture('IMG_2487.MOV')
#video_capture = skvideo.io.vread('IMG_2487.MOV')


while True:
    _, frame= video_capture.read()
    cv2.flip(frame, flipCode=-1)
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas= detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()