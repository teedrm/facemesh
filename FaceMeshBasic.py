import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture("video/1.mp4")
# previousTime = 0

# mpDraw = mp.solutions.drawing_utils
# mpFaceMesh = mp.solutions.face_mesh
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

# #change thickness of line/circles


# while True:
#     success, img = cap.read()
#     #find different points on face
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)
    
#     #display results - when something is dectected -> draw
#     if results.multi_face_landmarks:
#         #loop through the faces if there's more than 1
#         for faceLms in results.multi_face_landmarks:
#             mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

#     currentTime = time.time()
#     fps = 1/(currentTime-previousTime)
#     previousTime = currentTime
#     #20, 7 = location
#     #3, (0,255,0), 3 = scale, colour(green), thickness

#     cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
#                 3, (0, 255, 0), 3)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()