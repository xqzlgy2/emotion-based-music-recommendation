import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def generate_dataset(path):
    df = pd.read_csv(path)

    # remove unused columns
    unused_features = ["artists", "explicit", "id", "name", "release_date"]
    df = df.drop(unused_features, axis=1)

    x = df.drop(["majority_genre", "minority_genre"], axis=1)
    y = df["majority_genre"]

    feature_list = x.keys().to_list()
    for idx in range(len(feature_list)):
        print(str(idx) + ': ' + feature_list[idx])

    return x.values, y.values


def train_xgb_model(x, y):
    num_classes = len(set(y))
    model = xgb.XGBClassifier(learning_rate=0.3,
                              max_depth=6,
                              n_estimators=100,
                              objective='multi:softmax',
                              num_classes=num_classes)

    accuracy = cross_val_score(model, x, y, cv=10, scoring='accuracy').mean()
    f1 = cross_val_score(model, x, y, cv=10, scoring='f1_macro').mean()
    print('averaged accuracy = {:.2f}, F1 score = {:.2f}\n'.format(accuracy, f1))

    # show feature importance and save model
    model.fit(x, y)
    plot_xgb_feature_importance(model)
    with open('../models/genre_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)


def plot_xgb_feature_importance(model):
    plot_importance(model)
    pyplot.show()


def train_hierarchical_model(x, y):
    # binary classification for rock first, then other labels
    k_fold = KFold(n_splits=10)
    scores = []

    for idx_train, idx_val in k_fold.split(x, y):
        train_x = x[idx_train]
        train_y = y[idx_train]
        val_x = x[idx_val]
        val_y = y[idx_val]
        pred_y = []

        binary_model = train_binary_model(train_x, train_y)
        multi_model = train_multi_model(train_x, train_y)

        for feature in val_x:
            pred = hierarchical_predict(binary_model, multi_model, feature)
            pred_y.append(pred[0])
        pred_y = np.array(pred_y)
        hierarchical_score = accuracy_score(val_y, pred_y)

        filtered_val_x, filtered_val_y = filter_data(val_x, val_y)
        print('binary-model score: ' + str(binary_model.score(val_x, to_binary_label(val_y))))
        print('multi-model score: ' + str(multi_model.score(filtered_val_x, filtered_val_y)))
        print('hierarchical-model score: ' + str(hierarchical_score))

    print(np.mean(scores))


def train_binary_model(x, y):
    binary_y = to_binary_label(y)
    model = xgb.XGBClassifier(learning_rate=0.3,
                              max_depth=6,
                              n_estimators=100,
                              objective='binary:logistic')
    model.fit(x, binary_y)
    return model


def to_binary_label(y):
    binary_y = list(map(lambda label: 0 if label == 'Rock' else 1, y))
    return np.array(binary_y)


def train_multi_model(x, y):
    filtered_x, filtered_y = filter_data(x, y)
    num_classes = len(set(filtered_y))
    model = xgb.XGBClassifier(learning_rate=0.3,
                              max_depth=6,
                              n_estimators=100,
                              objective='multi:softmax',
                              num_classes=num_classes)
    model.fit(filtered_x, filtered_y)
    return model


def filter_data(x, y):
    filtered_x, filtered_y = [], []

    for idx in range(len(x)):
        if y[idx] == 'Rock':
            continue
        filtered_x.append(x[idx])
        filtered_y.append(y[idx])

    return np.array(filtered_x), np.array(filtered_y)


def hierarchical_predict(binary_model, multi_model, features):
    binary_result = binary_model.predict(np.array([features]))

    # if predicted as Rock, finish
    if binary_result == 0:
        return np.array(["Rock"])

    # otherwise use multi model do prediction
    return multi_model.predict(np.array([features]))


def classify():
    data_path = os.path.join('data', 'processed_data.csv')
    x, y = generate_dataset(data_path)
    train_xgb_model(x, y)
    # train_hierarchical_model(x, y)


def load_model():
    with open('../models/genre_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
        return model


if __name__ == '__main__':
    classify()
