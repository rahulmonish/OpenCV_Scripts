#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:15:15 2020

@author: rahul
"""

import cv2

input1= cv2.imread('cat.jpg')

#Show the image to the user
cv2.imshow('Test Cat Image', input1)

#wait for the image to be clear.
cv2.waitKey(2)

#This closes all the windows or else it will hang
cv2.destroyAllWindows()



#-----------------------------------------------#

import numpy as np

#Get all features of the face from the xml file into the face_classifier object
face_classifier= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Load the image
image= cv2.imread('IMG_6186.jpg')

#Convert the image to greyscale. By default, opencv reads an image in BGR and not RGB.
#The reason for converting to grey scale is for quicker processing
#since only one channel is involved
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces= face_classifier.detectMultiScale(gray, 1.3, 5)

if faces is ():
    print('No face found')


#For detecting all faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    print(x,y,w,h)
    #cv2.imshow('Face Detection', image)
    cv2.waitKey(2)
    cv2.destroyAllWindows()



#XML file to detect Eye
eye_classifier= cv2.CascadeClassifier('haarcascade_eye.xml')

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    print(x,y,w,h)
    #cv2.imshow('Face Detection', image)
    roi_grey= gray[y:y+h, x:x+w]
    roi_color= image[y:y+h, x:x+w]
    eyes= eye_classifier.detectMultiScale(gray)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(image, (ex,ey), (ex+ew, ey+eh), (255,255,0), 2)
        cv2.imshow('img',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




