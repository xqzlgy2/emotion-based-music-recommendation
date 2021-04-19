import base64
import json
import logging
import os
import random
import uuid
from sys import stdout
from typing import Counter

import cv2
import numpy as np
import spotipy
import torch
from flask import Flask, render_template, session, request, redirect, jsonify
from flask_session.__init__ import Session
from flask_socketio import SocketIO, emit
from scipy.special import softmax

from config import expressions, net, transform_image, detector, predictor, transform_image_shape_no_flip, \
    SpotifyCacheAuth, recorded_data, emotion_cache_folder, clear_recorded_data, caches_folder, target_length
from data_analyze.classifier import load_model
from utils import readb64, normalize

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
        # return sign in link if no token provided
        auth_url = spotifyCacheAuth.auth_manager.get_authorize_url()
        return jsonify(data={'content': auth_url, 'isLogin': False})

    # If signed in, return user name
    spotify = spotipy.Spotify(auth_manager=spotifyCacheAuth.auth_manager)
    return jsonify(data={'content': spotify.me(), 'isLogin': True})


@app.route('/sign_out')
def sign_out():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        spotifyCacheAuth = SpotifyCacheAuth()
        os.remove(spotifyCacheAuth.session_cache_path())
        os.remove(emotion_cache_folder + str(session.get('uuid')))
        session.clear()
        return jsonify(data={'success': True})

    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
        return jsonify(data={'success': False})


@app.route('/playlists')
def playlists():
    spotifyCacheAuth = SpotifyCacheAuth()
    if not spotifyCacheAuth.validate_token():
        return redirect('/')

    spotify = spotipy.Spotify(auth_manager=spotifyCacheAuth.auth_manager)
    track_ids = get_playlists_tracks(spotify)
    results = get_playlists_features(spotify, track_ids)
    return jsonify(data=results)


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
    features, artists, genres = [], [], []

    # limitation for track info is 50
    for idx in range(0, len(track_ids), 50):
        end = min(idx + 50, len(track_ids))
        slice = track_ids[idx: end]
        audio_features = spotify.audio_features(slice)
        tracks_info = spotify.tracks(slice)['tracks']

        for i in range(len(audio_features)):
            info = tracks_info[i]

            # build feature vector for a track
            info['year'] = info['album']['release_date'].split('-')[0]
            audio_features[i].update(info)
            vec = build_vector(audio_features[i])
            features.append(vec)

            # record artists of tracks
            artists_ids = list(map(lambda x: x['id'], info['artists']))
            artists.extend(artists_ids)

    features = np.array(features)
    model = load_model('./models/genre_classifier.pkl')

    # avoid exception of no playlist exists
    if len(features) != 0:
        genres = model.predict(features)

    track_valences = list(map(lambda x: {track_ids[x[0]]: x[1][12]}, enumerate(features)))
    recommend_res = get_recommendation(spotify, track_valences, genres, artists)

    return recommend_res


def build_vector(feature_dict):
    vec = []
    fields = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
              'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo', 'valence', 'year']

    for field in fields:
        vec.append(float(feature_dict[field]))

    return np.array(vec)


def get_recommendation(spotify, track_valences, genres, artists):
    top_num = 5
    # analyze playlist info, get top genres
    genre_count = sorted(Counter(genres).items(), key=lambda x: -x[1])
    top_genres = genre_count[:min(len(genres), top_num)]

    # valences = list(map(lambda x: list(x.values())[0], track_valences))
    # median_valence = np.median(valences)
    user_valence = get_detected_emotion()
    target_valence = 1 - user_valence

    seed_genres = list(map(lambda x: x[0], top_genres))
    seed_spotify_genres = []

    # map dataset label to spotify genres, do random selection
    with open('./data_analyze/data/genre_mapping.json') as f:
        genre_map = json.load(f)
        for genre in seed_genres:
            seed_spotify_genres.extend(genre_map[genre])
    seed_spotify_genres = random.sample(seed_spotify_genres, min(top_num, len(seed_spotify_genres)))

    api_response = None
    while not api_response:
        try:
            api_response = spotify.recommendations(seed_genres=seed_spotify_genres, target_valence=target_valence,
                                                   limit=10)
        except spotipy.exceptions.SpotifyException:
            # if no song could be found, resample genres
            seed_spotify_genres = random.sample(seed_spotify_genres, min(top_num, len(seed_spotify_genres)))

    result = list(map(lambda x: extract_result_fields(x), api_response['tracks']))

    return {"songs": result, "detected_valence": round(user_valence, 3)}


def extract_result_fields(info):
    return {
        'name': info['name'],
        'artists': list(map(lambda x: x['name'], info['artists'])),
        'url': info['external_urls']['spotify']
    }


def get_detected_emotion():
    path = emotion_cache_folder + str(session.get('uuid'))

    # if no user file provided, use neutral instead
    if not os.path.isfile(path):
        return 0.5

    with open(path, 'r') as f:
        detected_emotion = json.load(f)
        return detected_emotion['avg_valence']


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
        image, landmarks = transform_image_shape_no_flip(frame, bb=bounding_box)
        image = np.ascontiguousarray(image)
        tensor = transform_image(image).reshape(1,3,256,256)
        tensor = tensor.to('cpu')
        with torch.no_grad():
            output = net(tensor)
            emotion_raw = output['expression'].detach().numpy()
            emotion_prob = softmax(emotion_raw)
            valence = normalize(float(output['valence'].detach().numpy()[0]))
            arousal = normalize(float(output['arousal'].detach().numpy()[0]))
            results = {"emotion": expressions[np.argmax(emotion_prob)],
                       "valence": str(round(valence, 4)),
                       "arousal": str(round(arousal, 4))}

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
    # openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout key.pem -out cert.pem
    socketio.run(app, host='0.0.0.0', certfile='/etc/letsencrypt/live/emotionbasedmusicrecommendation.xyz/cert.pem',
                 keyfile='/etc/letsencrypt/live/emotionbasedmusicrecommendation.xyz/privkey.pem')
