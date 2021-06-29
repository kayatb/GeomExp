# ==============================================================================
# Functions for initializing the DNN models with their optimal found 
# hyperparameter settings.
# ==============================================================================

import numpy as np   # https://numpy.org
import tensorflow as tf  # https://www.tensorflow.org


def init_model_S():
    """ Return the model with the found optimal parameters for the S set. """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(352, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(7, activation='softmax')  # Amount of classes
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def init_model_HL():
    """ Return the model with the found optimal parameters for the HR set. """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(7, activation='softmax')  # Amount of classes
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def init_model_HR():
    """ Return the model with the found optimal parameters for the HL set. """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(352, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(7, activation='softmax')  # Amount of classes
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def calc_concatenated_acc(model_S, model_HL, model_HR, x_S, x_HL, x_HR, y_S, y_HL, y_HR):
    """ Calculates the accuracy of the concatenated results from the three given
    models. """
    y = np.concatenate([y_S, y_HL, y_HR])
    y_pred_S = model_S.predict_classes(x_S, verbose=1)
    y_pred_HL = model_HL.predict_classes(x_HL, verbose=1)
    y_pred_HR = model_HR.predict_classes(x_HR, verbose=1)
    y_preds = np.concatenate([y_pred_S, y_pred_HL, y_pred_HR])

    correct = 0
    for y_true, y_pred in zip(y, y_preds):
        if y_true == y_pred:
            correct += 1

    return correct / len(y)
