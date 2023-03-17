import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/1.mp4")
previousTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

#change thickness of line/circles


while True:
    success, img = cap.read()
    #find different points on face
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    #display results - when something is dectected -> draw
    if results.multi_face_landmarks:
        #loop through the faces if there's more than 1
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    #20, 7 = location
    #3, (0,255,0), 3 = scale, colour(green), thickness

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)