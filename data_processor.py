import os
import ast
import time
import h5py
import pandas as pd


def save_labeled_results(output_path, names, titles, majority_genres, minority_genres):
    dataframe = pd.DataFrame({'artist_name': names, 'title': titles,
                              'majority_genre': majority_genres, 'minority_genre': minority_genres})
    # remove previous results
    if os.path.isfile(output_path):
        os.remove(output_path)
    dataframe.to_csv(output_path, index=False, sep=',')


def combine_label_msd(cls_path, msd_path, output_path):
    # size of million song dataset is around 300GB. It could be accessed through AWS snapshot.
    # snapshot ID: snap-5178cf30, EC2 region should be us-east-1

    names, titles, majority_genres, minority_genres = [], [], [], []
    start_time = time.time()

    # read label file line by line
    with open(cls_path) as f:
        for line in f:
            # skip line of comments
            if line.startswith('#'):
                continue

            attrs = line.split()
            track_id = attrs[0]
            majority_genre = attrs[1]
            minority_genre = attrs[2] if len(attrs) > 2 else ''

            track_info = get_track_features(track_id, msd_path)
            if not track_info:
                continue

            names.append(track_info[0])
            titles.append(track_info[1])
            majority_genres.append(majority_genre)
            minority_genres.append(minority_genre)

            if len(names) % 1000 == 0:
                print(str(len(names)) + ' row processed. Time used: ' + str(time.time() - start_time) + 's')

                # save intermediate results to avoid data loss
                if len(names) % 10000 == 0:
                    save_labeled_results(output_path, names, titles, majority_genres, minority_genres)

    save_labeled_results(output_path, names, titles, majority_genres, minority_genres)


def get_track_features(track_id, msd_path):
    # extract fields from h5 file
    h5_path = os.path.join(msd_path, track_id[2], track_id[3], track_id[4], track_id + '.h5')

    if not os.path.isfile(h5_path):
        return None
    track_features = h5py.File(h5_path, 'r')

    artist_name = track_features['metadata']['songs']['artist_name'][0].decode('UTF-8')
    title = track_features['metadata']['songs']['title'][0].decode('UTF-8')

    return artist_name, title


def filter_title(name, title):
    if name.startswith(title) or title.startswith(name) or name.endswith(title) or title.endswith(name):
        return True
    return False


def filter_artist(artists, target):
    artists_list = ast.literal_eval(artists)
    for artist in artists_list:
        if artist in target:
            return True
    return False


def save_combined_results(res_dfs, output_path):
    combined_res = pd.concat(res_dfs, axis=0)
    combined_res.to_csv(output_path, index=False, sep=',')


def combine_label_features(feature_path, label_path, output_path):
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    res_dfs = []
    start_time = time.time()

    for idx, row in labels.iterrows():
        if idx <= 100000:
            continue
        artist = row['artist_name']
        title = row['title']

        # filter by song name first, then by artist
        candidates = features[features.name.apply(lambda x: filter_title(x, title))]
        res = candidates[candidates.artists.apply(lambda x: filter_artist(x, artist))]

        if idx % 100 == 0:
            print(str(idx) + ' row processed. ' + 'Time used: ' + str(time.time() - start_time))
        # print(artist + ',\t' + title + '\t')
        # print(candidates)
        # print(res)

        res.insert(res.shape[1], 'majority_genre', row['majority_genre'])
        res.insert(res.shape[1], 'minority_genre', row['minority_genre'])
        res_dfs.append(res)

        if idx % 1000 == 0:
            save_combined_results(res_dfs, output_path)

    save_combined_results(res_dfs, output_path)


if __name__ == '__main__':
    cls_file_path = os.path.join('data', 'msd_tagtraum_cd2.cls')
    msd_folder_path = os.path.join('data', 'millionsongsubset')
    label_output_path = os.path.join('data', 'labeled_songs.csv')
    spotify_feature_path = os.path.join('data', 'spotify_features', 'data.csv')
    result_path = os.path.join('data', 'processed_data.csv')

    # combine_label_msd(cls_file_path, msd_folder_path, label_output_path)
    combine_label_features(spotify_feature_path, label_output_path, result_path)
