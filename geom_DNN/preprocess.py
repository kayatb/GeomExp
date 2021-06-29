# ==============================================================================
# Functions for preprocessing the data from csv-files into datastructures useful
# for further processing, classification, etc.
# ==============================================================================

import pandas as pd  # https://pandas.pydata.org
from sklearn.preprocessing import StandardScaler  # https://scikit-learn.org/stable/


def load_data(file_name, per_pose=False):
    """ Load data from the given csv-file and can split on pose. """
    df = pd.read_csv(file_name, header=0)

    if per_pose:
        # Split the dataset per pose (S/Frontal, Half Left, Half Right).
        # Extract the pose from the file name/id.
        mask1 = df['Filename'].str.endswith('S.JPG')
        mask2 = df['Filename'].str.endswith('HL.JPG')
        mask3 = df['Filename'].str.endswith('HR.JPG')

        dfS = df[mask1]
        dfHL = df[mask2]
        dfHR = df[mask3]

        return (dfS, dfHL, dfHR)

    else:  # No splitting on poses.
        return df


def get_x_and_y(df):
    """ Split a dataframe into x (the features) and y (the label). """
    label_nums = {'AF': 0, 'AN': 1, 'DI': 2, 'HA': 3, 'NE': 4, 'SA': 5, 'SU': 6}

    x = df[df.columns[1:]]  # Columns that contain the features.
    y = df['Filename'].apply(lambda x: label_nums[x[4:6]])  # Extract the label from the ID.
    y = y.to_numpy()  # Convert to a simple 1d numpy array.

    return (x, y)


def standardize_data(x_train, x_test):
    """ Standardize the data by fitting a scaler to the train data. """
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_stand = scaler.transform(x_train)
    x_test_stand = scaler.transform(x_test)

    return (x_train_stand, x_test_stand)
