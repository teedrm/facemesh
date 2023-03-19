import cv2
import numpy as np
import dlib


detector = dlib.get_frontal_face_detector()

#detect landmark for the face - need predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#function to crop img - bounding box around a feature that want to crop
def createBox(img, points):
    bbox = cv2.boundingRect(points)
    x,y,w,h = bbox
    imgCrop = img[y:y+h, x:x+w]
    # imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
    return imgCrop

img = cv2.imread('./img/1.jpg')
# img = cv2.resize(img, (0,0), None, 0.5, 0.5)
imgOriginal = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)
#store wanted points
myPoints = []

for face in faces:
    x1,y1 = face.left(), face.top()
    x2,y2 = face.right(), face.bottom()
    imgOriginal = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255,0),2)
    #give img + face object - find all the landmarks on img
    landmarks= predictor(imgGray, face)
    #draw - to find the feature
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x,y])
        cv2.circle(imgOriginal, (x,y), 2, (50, 50, 255), cv2.FILLED)
        #getting the coors from landmarks
        cv2.putText(imgOriginal,str(n),(x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.3, (0,0,255), 1)

        #converting list to numpy array
        myPoints = np.array(myPoints)
        imgLeftEye = createBox(img, myPoints[36:42])
        cv2.imshow('LeftEye', imgLeftEye)
    print(myPoints)

cv2.imshow("Original", imgOriginal)
cv2.waitKey(0)