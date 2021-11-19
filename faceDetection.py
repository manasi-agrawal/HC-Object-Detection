import cv2
import numpy as np
img = cv2.imread('faces.jpg')
#print(img.shape)

print('orignal dimensions :', img.shape)
scale_percent = 40
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
#resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

#('resized image:', resized.shape)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)

#if faces is ():
  #  print("no face found")
for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (127, 0, 255), 2)
    cv2.imshow('face Detection', resized)
    cv2.waitKey(0)
cv2.destroyAllWindows()

