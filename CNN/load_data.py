# ==============================================================================
# Several functions to load image data into different datastructures.
# ==============================================================================

import tensorflow as tf  # https://www.tensorflow.org
import os
import numpy as np  # https://numpy.org


def resize_images(input_path, save_path):
    """ Resize images to 224 x 224 format. """
    from PIL import Image

    for file in sorted(os.listdir(input_path)):
        im = Image.open(os.path.join(input_path, file))
        out = im.resize((224, 224))
        out.save(os.path.join(save_path, file))


def load_dataset(file_path):
    """ Get the x and y data set. x consists of the images and y of the
    corresponding labels. Loaded into a tensorflow dataset. """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(file_path, label_mode='int', image_size=(224, 224))

    return dataset


def load_data_np(file_path, file_names):
    """ Load the data into a numpy array. """
    from PIL import Image

    labels = []
    class_names = {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'NE': 4, 'SA': 5, 'SU': 6}

    all_files = [os.path.join(file_path, f) for f in file_names]

    labels = np.array([class_names[f[4:6]] for f in file_names])
    x = np.array([np.array(Image.open(f), dtype=np.float) for f in all_files])

    return x, labels
