import json
import random
import warnings
import spotipy
import requests
import csv
import pandas as pd
import spotipy
from dateutil.parser import parse as parse_date
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth

scope = "user-library-read"

# try to get the token
credentials = oauth2.SpotifyClientCredentials(client_id="8223aaebb72447d9b099839baef55558", client_secret="1e55403cf33c48d58f636fe8b9a54581")
sp = spotipy.Spotify(auth_manager=credentials)

song_name = []
song_attribute = []

token = credentials.get_access_token()
headers = {"Authorization": "Bearer {}".format(token)}

# playlist Front Left   https://open.spotify.com/playlist/37i9dQZF1DX5WTH49Vcnqp?si=eEa5W_dxTNS1_YdAxS-hHw
# playlist Alan walker  https://open.spotify.com/playlist/37i9dQZF1DX4npDJDFDYLg?si=GfdNGiAWQsCY24d7ra1ABw



# song's attribute Front Left
playlist = sp.playlist("37i9dQZF1DX5WTH49Vcnqp")

tracks_df = pd.DataFrame([(track['track']['id'],
                           track['track']['artists'][0]['name'],
                           track['track']['album']['name'],
                           track['track']['disc_number'],
                           track['track']['track_number'],
                           track['track']['name'],
                           parse_date(track['track']['album']['release_date']) if track['track']['album']['release_date'] else None,
                           parse_date(track['added_at']))
                          for track in playlist['tracks']['items']],
                         columns=['id', 'artist', 'album','disc','track_number','name', 'release_date', 'added_at'] )

tracks_df['target'] = 'A'
tracks_df.head(100)

# write to csv
tracks_df.to_csv("Front_Left.csv")

# get song's attribute
columns = ["danceability","energy","key", "loudness", "mode"]

with open("attribute.csv","w",newline = "") as csvfile:
    write_csv = csv.writer(csvfile)
    write_csv.writerow(["danceability","energy","key", "loudness", "mode","speechiness","acousticness","instrumentalness",
                        "liveness","valence","tempo","type","id", "uri","track_href","analysis_url","duration_ms","time_signature"])

    for track in playlist['tracks']['items']:
        id = track["track"]["id"]
        song_attributes = requests.get(f"https://api.spotify.com/v1/audio-features/{id}", headers=headers)
        song_attr_dict = song_attributes.json()
        write_csv.writerow([song_attr_dict["danceability"], song_attr_dict["energy"], song_attr_dict["key"] ,song_attr_dict["loudness"], song_attr_dict["mode"], song_attr_dict["speechiness"]
                             ,song_attr_dict["acousticness"],song_attr_dict["instrumentalness"],song_attr_dict["liveness"],song_attr_dict["valence"],song_attr_dict["tempo"],song_attr_dict["type"]
                             ,song_attr_dict["id"],song_attr_dict["uri"],song_attr_dict["track_href"],song_attr_dict["analysis_url"],song_attr_dict["duration_ms"],song_attr_dict["time_signature"]])


playlist = sp.playlist("37i9dQZF1DX5WTH49Vcnqp")

tracks_df = pd.DataFrame([(track['track']['id'],
                           track['track']['artists'][0]['name'],
                           track['track']['album']['name'],
                           track['track']['disc_number'],
                           track['track']['track_number'],
                           track['track']['name'],
                           parse_date(track['track']['album']['release_date']) if track['track']['album']['release_date'] else None,
                           parse_date(track['added_at']))
                          for track in playlist['tracks']['items']],
                         columns=['id', 'artist', 'album','disc','track_number','name', 'release_date', 'added_at'] )

tracks_df['target'] = 'A'
tracks_df.head(100)

# write to csv
tracks_df.to_csv("Front_Left.csv")

# get song's attribute
columns = ["danceability","energy","key", "loudness", "mode"]

with open("attribute.csv","w",newline = "") as csvfile:
    write_csv = csv.writer(csvfile)
    write_csv.writerow(["danceability","energy","key", "loudness", "mode","speechiness","acousticness","instrumentalness",
                        "liveness","valence","tempo","type","id", "uri","track_href","analysis_url","duration_ms","time_signature"])

    for track in playlist['tracks']['items']:
        id = track["track"]["id"]
        song_attributes = requests.get(f"https://api.spotify.com/v1/audio-features/{id}", headers=headers)
        song_attr_dict = song_attributes.json()
        write_csv.writerow([song_attr_dict["danceability"], song_attr_dict["energy"], song_attr_dict["key"] ,song_attr_dict["loudness"], song_attr_dict["mode"], song_attr_dict["speechiness"]
                             ,song_attr_dict["acousticness"],song_attr_dict["instrumentalness"],song_attr_dict["liveness"],song_attr_dict["valence"],song_attr_dict["tempo"],song_attr_dict["type"]
                             ,song_attr_dict["id"],song_attr_dict["uri"],song_attr_dict["track_href"],song_attr_dict["analysis_url"],song_attr_dict["duration_ms"],song_attr_dict["time_signature"]])







