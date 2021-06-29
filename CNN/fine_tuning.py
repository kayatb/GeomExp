# ==============================================================================
# Fine-tune a pre-trained CNN model on new data.
# ==============================================================================

import tensorflow as tf  # https://www.tensorflow.org
from PIL import Image  # https://pillow.readthedocs.io/en/stable/
import numpy as np  # https://numpy.org
from typing import Tuple, List
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_vggface import utils  # https://github.com/rcmalli/keras-vggface
import os
import tempfile


def load_model(file_path):
    """ Load the pre-trained model from json-file
    and the weights from h5-file. """
    with open(os.path.join(file_path, 'model.json'), 'r') as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(os.path.join(file_path, 'weights.h5'))

    model.compile(metrics=['accuracy'], loss='sparse_categorical_crossentropy')

    return model

    
def preprocess_image(image: np.ndarray):
    """ Output: np.ndarray
    There is special function in keras_vggface to preprocess input img to the
    format applicable to the VggFace2 model: utils.preprocess_input(image, version=2)
    Basically, they just substract mean calculated on their own dataset. """
    if image.shape[0] != 224:
        raise Exception("image shape should be 224x224x3")

    image = utils.preprocess_input(image, version=2)

    return image


def load_image(path: str, needed_size: Tuple[int, int]):
    """ Output: np.ndarray
    We load image by PIL.Image lib. The resize procedure is also included in this
    function """
    img = Image.open(path)
    img = img.resize(needed_size)
    img = np.array(img)

    return img


def load_preprocess_image(path: str, needed_size: Tuple[int, int]):
    """ Output: np.ndarray
    just to combine two aforementioned functions. """
    img = load_image(path, needed_size)
    img = preprocess_image(img)

    return img


def get_model_prediction(model: tf.keras.Model, img: np.ndarray):
    """ Output: np.ndarray
    We can predict either one image or batch of images.
    For example, if you feed into this function 5 images, the input array will
    be (5,224,224,30), and the prediction from model will be (5,7) 7 - number of emotions
    if the image will be (224,224,3), we will transform it to (1,224,224,3) and then
    you will get (1,7) """
    if len(img.shape) == 3:
        img = img[np.newaxis, ...]

    prediction = model.predict(img)

    return prediction


def freeze_n_layers(model: tf.keras.Model, n: int):
    """ Output: tf.keras.Model
    Function to freeze first N layers. """
    for i in range(n):
        model.layers[i].trainable = False

    return model


def freeze_layers_by_indices(model: tf.keras.Model, indices: List[int]):
    """ Output: tf.keras.Model
    The same thing as above, but now indexes of layers are provided. """
    for idx in indices:
        model.layers[idx].trainable = False

    return model

# remember, you should execute model.compile(...) after you freeze the layers


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.01)):
    """ Add regularisation to the model layers.
    source: https://sthalles.github.io/keras-regularizer/ """
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


if __name__ == '__main__':
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=50,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator()

    # Load the data.
    generator = train_datagen.flow_from_directory("PATH/TO/TRAIN_IMG", class_mode='sparse')
    gen_val = val_datagen.flow_from_directory("PATH/TO/VAL_IMG", class_mode='sparse')

    # Load the base model.
    model = load_model("PATH/TO/BASE_MODEL")  # Has 178 layers (in my case).
    model.load_weights("PATH/TO/WEIGHTS")
    model = add_regularization(model)

    # Freeze layers.
    model = freeze_n_layers(model, 175)  # Freeze every layer but the top.

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(metrics=['accuracy'], optimizer=optimizer, loss='sparse_categorical_crossentropy')

    model.fit(generator, epochs=50,
              validation_data=gen_val,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    tf.saved_model.save(model, "PATH/TO/SAVE/MODEL")
