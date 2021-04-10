import os
import base64
import logging
from sys import stdout

import cv2
import dlib
import numpy as np
import torch
from flask import Flask, render_template, Response, session, request, redirect
from flask_session.__init__ import Session
import spotipy
import uuid
from camera import VideoCamera
from spotipy.oauth2 import SpotifyOAuth

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
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

def session_cache_path():
    return caches_folder + session.get('uuid')

video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

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
    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port="5000")
