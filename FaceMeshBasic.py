import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/1.mp4")
previousTime = 0
while True:
    success, img = cap.read()
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    #20, 7 = location
    #3, (0,255,0), 3 = scale, colour(green), thickness

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)