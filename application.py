import base64
import logging
import os
import uuid
from sys import stdout

import cv2
import numpy as np
import spotipy
import torch
from flask import Flask, render_template, session, request, redirect
from flask_session.__init__ import Session
from flask_socketio import SocketIO, emit
from scipy.special import softmax

from config import expressions, net, transform_image, detector, predictor, transform_image_shape_no_flip, \
    SpotifyCacheAuth, session_cache_path
from utils import readb64

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
socketio = SocketIO(app)

app.config['DEBUG'] = False
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)


@app.route('/')
def index():
    if request.args.get("code"):
        # Step 3. Being redirected from Spotify auth page
        spotifyCacheAuth = SpotifyCacheAuth()
        token = spotifyCacheAuth.auth_manager.get_access_token(request.args.get("code"))
        spotifyCacheAuth.cache_handler.save_token_to_cache(token)
    return render_template('index.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/sign_in')
def sign_in():
    if not session.get('uuid'):
        # Step 1. Visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())
    spotifyCacheAuth = SpotifyCacheAuth()
    if not spotifyCacheAuth.validate_token():
        # Step 2. Display sign in link when no token
        auth_url = spotifyCacheAuth.auth_manager.get_authorize_url()
        return f'<h2><a href="{auth_url}">Sign in</a></h2>'
        # return render_template('index.html', auth_url=auth_url)

    # Step 4. Signed in, display data
    spotify = spotipy.Spotify(auth_manager=spotifyCacheAuth.auth_manager)
    return f'<h2>Hi {spotify.me()["display_name"]}, ' \
           f'<small><a href="/sign_out">[sign out]<a/></small></h2>' \
           f'<a href="/playlists">Allow access to my playlists</a> '


@app.route('/sign_out')
def sign_out():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        os.remove(session_cache_path())
        session.clear()
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    return redirect('/')


@app.route('/playlists')
def playlists():
    spotifyCacheAuth = SpotifyCacheAuth()
    if not spotifyCacheAuth.validate_token():
        return redirect('/')

    spotify = spotipy.Spotify(auth_manager=spotifyCacheAuth.auth_manager)
    requestResponse = spotify.current_user_playlists()
    if requestResponse is not None:
        return requestResponse


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


if __name__ == '__main__':
    socketio.run(app)
