import cv2
import numpy as np
import dlib

img = cv2.imread('./img/image.jpg')
# img = cv2.resize(img, (0,0), None, 0.5, 0.5)
imgOriginal = img.copy()

detector = dlib.get_frontal_face_detector()

#detect landmark for the face - need predictor

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
    x1,y1 = face.left(), face.top()
    x2,y2 = face.right(), face.bottom()
    imgOriginal = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255,0),2)

cv2.imshow("Original", imgOriginal)
cv2.waitKey(0)