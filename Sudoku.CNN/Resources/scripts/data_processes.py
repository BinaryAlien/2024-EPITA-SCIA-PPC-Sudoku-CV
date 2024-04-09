import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def get_data_one_shot_fill(file):

    data = pd.read_csv(file)#.sample(n=10000)
    #print(data.head)

    feat_raw = data['quizzes']
    label_raw = data['solutions']

    feat = []
    label = []

    for i in feat_raw:

        x = np.array([int(j) for j in i]).reshape((9,9,1))
        feat.append(x)

    feat = np.array(feat)
    feat = feat/9
    feat -= .5

    for i in label_raw:
        x = np.array([int(j) for j in i]) - 1  # Zero-indexing the labels
        x = to_categorical(x, num_classes=9)  # Convert labels to categorical

        # Reshape each label to (9,9,9) to reflect the Sudoku's 9 boxes of 3x3
        x_reshaped = x.reshape((9,9,9))
        label.append(x_reshaped)

    label = np.array(label)  # This will have the shape (-1, 9, 9, 9)



    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def get_data(file):

    data = pd.read_csv(file)#.sample(n=100000)
    #print(data.head)

    feat_raw = data['quizzes']
    label_raw = data['solutions']

    feat = []
    label = []

    for i in feat_raw:

        x = np.array([int(j) for j in i]).reshape((9,9,1))
        feat.append(x)

    feat = np.array(feat)
    feat = feat/9
    feat -= .5

    for i in label_raw:

        x = np.array([int(j) for j in i]).reshape((81,1)) - 1
        label.append(x)

    label = np.array(label)

    del(feat_raw)
    del(label_raw)

    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test