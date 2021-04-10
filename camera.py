import cv2
import dlib
from deepface import DeepFace
from matplotlib import pyplot as plt
from collections import defaultdict

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv2.VideoCapture(0)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.recorded_emotion = defaultdict(list)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        obj = DeepFace.analyze(img_path=frame, actions=['emotion'])
        print(obj['emotion'])
        for k, v in obj['emotion'].items():
            self.recorded_emotion[k].append(v)

        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
