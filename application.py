import base64
import json
import logging
import os
from typing import Counter
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
    SpotifyCacheAuth, recorded_data, emotion_cache_folder, clear_recorded_data, caches_folder, target_length
from utils import readb64
from data_analyze.classifier import load_model

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
socketio = SocketIO(app)

app.config['DEBUG'] = False
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

if not os.path.exists(emotion_cache_folder):
    os.makedirs(emotion_cache_folder)


@app.route('/')
def index():
    if request.args.get("code"):
        # if signed in, cache token
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
        # visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())
    spotifyCacheAuth = SpotifyCacheAuth()
    if not spotifyCacheAuth.validate_token():
        # display sign in link if no token provided
        auth_url = spotifyCacheAuth.auth_manager.get_authorize_url()
        return f'<h2><a href="{auth_url}">Sign in</a></h2>'

    # If signed in, display data
    spotify = spotipy.Spotify(auth_manager=spotifyCacheAuth.auth_manager)
    return f'<h2>Hi {spotify.me()["display_name"]}, ' \
           f'<small><a href="/sign_out">[sign out]<a/></small></h2>'


@app.route('/sign_out')
def sign_out():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        spotifyCacheAuth = SpotifyCacheAuth()
        os.remove(spotifyCacheAuth.session_cache_path())
        os.remove(emotion_cache_folder + str(session.get('uuid')))
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
    track_ids = get_playlists_tracks(spotify)
    features = get_playlists_features(spotify, track_ids)
    return str(features)


def get_playlists_tracks(spotify):
    requestResponse = spotify.current_user_playlists()
    track_ids = set()

    if requestResponse is not None:
        playlists = requestResponse["items"]

        for playlist in playlists:
            playlist_info = spotify.playlist(playlist["id"])
            tracks = playlist_info['tracks']['items']
            for track in tracks:
                track_ids.add(track["track"]["id"])

    return track_ids


def get_playlists_features(spotify, track_ids):
    track_ids = list(track_ids)
    features = []

    for idx in range(0, len(track_ids), 50):
        end = min(idx+50, len(track_ids))
        slice = track_ids[idx: end]
        audio_features = spotify.audio_features(slice)
        tracks_info = spotify.tracks(slice)['tracks']

        for i in range(len(audio_features)):
            tracks_info[i]['year'] = tracks_info[i]['album']['release_date'].split('-')[0]
            audio_features[i].update(tracks_info[i])
            vec = build_vector(audio_features[i])
            features.append(vec)

    features = np.array(features)
    model = load_model('./models/genre_classifier.pkl')
    results = model.predict(features)
    print([str(k) + ' ' + str(v/len(results)) for k,v in Counter(results).items()])
    return results


def build_vector(feature_dict):
    vec = []
    fields = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
             'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo', 'valence', 'year']
    
    for field in fields:
        vec.append(float(feature_dict[field]))

    return np.array(vec)


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@socketio.on('input image', namespace='/test')
def test_message(data):
    if session.get('uuid') not in recorded_data or recorded_data[session.get('uuid')] == {}:
        recorded_data[session.get('uuid')] = {"valence": [], "arousal": []}

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
            valence = float(output['valence'].detach().numpy()[0])
            arousal = float(output['arousal'].detach().numpy()[0])
            results = {"emotion": "Emotion: "+expressions[np.argmax(emotion_prob)],
                       "valence": "Valence: "+str(round(valence, 4)),
                       "arousal": "Arousal: "+str(round(arousal, 4))}

            recorded_data[session.get('uuid')]['valence'].append(valence)
            recorded_data[session.get('uuid')]['arousal'].append(arousal)

            ret, buffer = cv2.imencode('.png', frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            emit('out-image-event', {'image': image_data, 'results': results}, namespace='/test')

    if len(recorded_data[session.get('uuid')]['valence']) == target_length:
        with open(emotion_cache_folder + str(session.get('uuid')), 'w') as f:
            avg_valence = sum(recorded_data[session.get('uuid')]['valence']) / target_length
            avg_arousal = sum(recorded_data[session.get('uuid')]['arousal']) / target_length
            json.dump({"avg_valence": avg_valence, "avg_arousal": avg_arousal}, f)
            clear_recorded_data(session.get('uuid'))
        emit('finished-capturing')


if __name__ == '__main__':
    socketio.run(app)
