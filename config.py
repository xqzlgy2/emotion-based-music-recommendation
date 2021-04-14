import dlib
import spotipy
import torch
from flask import session
from torchvision import transforms

from emonet.emonet.data_augmentation import DataAugmentor
from emonet.emonet.models import EmoNet

# START -- config for facial expression recognition --
emotion_cache_folder = ".emotion_caches/"
recorded_valence = []
recorded_arousal = []


def clear_recorded_data():
    global recorded_valence, recorded_arousal
    recorded_arousal = []
    recorded_valence = []


image_size = 256
expressions = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger',
               7: 'contempt',
               8: 'none'}
net = EmoNet(n_expression=8)
state_dict = torch.load("emonet/pretrained/emonet_8.pth", map_location='cpu')
net.load_state_dict(state_dict, strict=False)
net.eval()
transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# END ---------------------------------------------------

# START -- config for Spotipy --
caches_folder = './.spotify_caches/'
SPOTIPY_CLIENT_ID = 'dde4e2ccdb1a4498aef96198f319a1e8'
SPOTIPY_CLIENT_SECRET = '8f9c59120ab949828c5936c751878797'
SPOTIPY_REDIRECT_URI = 'http://localhost:5000'


class SpotifyCacheAuth:

    def session_cache_path(self):
        return caches_folder + str(session.get('uuid'))

    def __init__(self):
        self.cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=self.session_cache_path())
        self.auth_manager = spotipy.oauth2.SpotifyOAuth(
            scope='playlist-read-private playlist-modify-private app-remote-control user-read-currently-playing',
            cache_handler=self.cache_handler,
            show_dialog=True,
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI)

    def validate_token(self):
        return self.auth_manager.validate_token(self.cache_handler.get_cached_token())
# END ---------------------------------------------------
