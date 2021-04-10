import os
import base64
import logging
from sys import stdout

import cv2
import dlib
import numpy as np
import torch
import spotipy
import uuid

from flask import Flask, render_template, Response, session, request, redirect
from flask_session.__init__ import Session
from flask_socketio import SocketIO, emit
from scipy.special import softmax
from torchvision import transforms
from spotipy.oauth2 import SpotifyOAuth

from emonet.emonet.data_augmentation import DataAugmentor
from emonet.emonet.models import EmoNet

#export SPOTIPY_CLIENT_ID='dde4e2ccdb1a4498aef96198f319a1e8'
#export SPOTIPY_CLIENT_SECRET='8f9c59120ab949828c5936c751878797'
#export SPOTIPY_REDIRECT_URI='http://localhost:5000/callback'

#scope = "user-library-read"
#sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

#playlists = sp.user_playlists('spotify')
#while playlists:
#	for i, playlist in enumerate(playlists['items']):
#		print("%4d %s %s" % (i + 1 +playlists['offset'], playlist['uri'], playlist['name']))
#	if playlists['next']:
#		playlists = sp.next(playlists)
#	else:
#		playlists = None

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
socketio = SocketIO(app)

app.config['DEBUG'] = False
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

# init model params
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


caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

def session_cache_path():
    return caches_folder + session.get('uuid')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/sign_in')
def sign_in():
    if not session.get('uuid'):
        # Step 1. Visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())

    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(scope = 'playlist-read-private playlist-modify-private app-remote-control user-read-currently-playing',
                                                cache_handler=cache_handler, 
                                                show_dialog=True,
                                                client_id = 'dde4e2ccdb1a4498aef96198f319a1e8',
                                                client_secret = '8f9c59120ab949828c5936c751878797',
                                                redirect_uri = 'http://localhost:5000')

    if request.args.get("code"):
        # Step 3. Being redirected from Spotify auth page
        auth_manager.get_access_token(request.args.get("code"))
        return redirect('/')
        #return render_template('index.html')


    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        # Step 2. Display sign in link when no token
        auth_url = auth_manager.get_authorize_url()
        return f'<h2><a href="{auth_url}">Sign in</a></h2>'
        #return render_template('index.html', auth_url=auth_url)

    # Step 4. Signed in, display data
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    return f'<h2>Hi {spotify.me()["display_name"]}, ' \
           f'<small><a href="/sign_out">[sign out]<a/></small></h2>' \
           f'<a href="/playlists">Allow access to my playlists</a> ' \
           #f'<a href="/currently_playing">currently playing</a> | ' \
		   #f'<a href="/current_user">me</a>' \
    #return render_template('index.html')


@app.route('/sign_out')
def sign_out():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        os.remove(session_cache_path())
        session.clear()
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    return redirect('/')
    #return render_template('index.html')


@app.route('/playlists')
def playlists():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler,
    											client_id = 'dde4e2ccdb1a4498aef96198f319a1e8',
                                                client_secret = '8f9c59120ab949828c5936c751878797',
                                                redirect_uri = 'http://localhost:5000')
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return redirect('/')
        #return render_template('index.html')

    spotify = spotipy.Spotify(auth_manager=auth_manager)
    requestResponse = spotify.current_user_playlists()
    if requestResponse is not None:
    	print (requestResponse)
    	return redirect('/')
    #return spotify.current_user_playlists()
    #print (spotify.current_user_playlists())
    #return render_template('index.html')


#@app.route('/currently_playing')
#def currently_playing():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler,
    											client_id = 'dde4e2ccdb1a4498aef96198f319a1e8',
                                                client_secret = '8f9c59120ab949828c5936c751878797',
                                                redirect_uri = 'http://localhost:5000')
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return redirect('/')
        #return render_template('index.html')
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    track = spotify.current_user_playing_track()
    if not track is None:
        return track
        #print (track)
        #return render_template('index.html')
    return "No track currently playing."
    #print ("No track currently playing.")
    #return render_template('index.html')


#@app.route('/current_user')
#def current_user():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler,
    											client_id = 'dde4e2ccdb1a4498aef96198f319a1e8',
                                                client_secret = '8f9c59120ab949828c5936c751878797',
                                                redirect_uri = 'http://localhost:5000')
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return redirect('/')
        #return render_template('index.html')
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    return spotify.current_user()
    #print (spotify.current_user())
    #return render_template('index.html')

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

if __name__ == '__main__':
    socketio.run(app)
