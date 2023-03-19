import cv2
import mediapipe as mp
import dlib
import numpy as np

img = cv2.imread("./img/image.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# We load Face detector and Face landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x,y))

        cv2.circle(img, (x, y) , 3, (0, 0, 255), -1)
cv2.imshow("Image 1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()