import base64
import logging
from sys import stdout

import cv2
import dlib
import numpy as np
import torch
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from scipy.special import softmax
from torchvision import transforms

from emonet.emonet.data_augmentation import DataAugmentor
from emonet.emonet.models import EmoNet

app = Flask(__name__)
app.config['DEBUG'] = False
app.logger.addHandler(logging.StreamHandler(stdout))
socketio = SocketIO(app)

image_size = 256
expressions = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt',
               8: 'none'}
net = EmoNet(n_expression=8)
state_dict = torch.load("emonet/pretrained/emonet_8.pth", map_location='cpu')
net.load_state_dict(state_dict, strict=False)
net.eval()
transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@socketio.on('input image', namespace='/test')
def test_message(data):
    frame = readb64(data['image'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        landmarks_raw = predictor(gray, rect)
        landmarks = []
        for n in range(0, 68):
            x = landmarks_raw.part(n).x
            y = landmarks_raw.part(n).y
            landmarks.append([x, y])
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        landmarks = np.array(landmarks)
        bounding_box = [landmarks.min(axis=0)[0], landmarks.min(axis=0)[1], landmarks.max(axis=0)[0],
                        landmarks.max(axis=0)[1]]
        frame, landmarks = transform_image_shape_no_flip(frame, bb=bounding_box)

        frame = np.ascontiguousarray(frame)
        tensor = transform_image(frame).reshape(1, 3, 256, 256)
        tensor = tensor.to('cpu')
        with torch.no_grad():
            output = net(tensor)
            emotion_raw = output['expression'].detach().numpy()
            emotion_prob = softmax(emotion_raw)

            results = {"emotion": expressions[np.argmax(emotion_prob)],
                       "valence": float(output['valence'].detach().numpy()[0]),
                       "arousal": float(output['arousal'].detach().numpy()[0])}

            ret, buffer = cv2.imencode('.png', frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            emit('out-image-event', {'image': image_data, 'results': results}, namespace='/test')


@app.route('/camera')
def index():
    return render_template('camera.html')


if __name__ == '__main__':
    socketio.run(app)
