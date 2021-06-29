# ==============================================================================
# Functions for performing hyperparameter optimization together with
# feature selection.
# ==============================================================================

import data
import feature_selection

import tensorflow as tf  # https://www.tensorflow.org
import kerastuner as kt  # https://keras.io/keras_tuner/


def build_model(hp):
    """ Build a model for hyperparameter tuning. """
    model = tf.keras.Sequential()

    for i in range(hp.Int('num_layers', 1, 4)):  # Variable amount of layers.
        # Variable amount of neurons and regularisation rate.
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('regul_' + str(i),
                                                                                    [1e-1, 1e-2, 1e-3]))
                                        ))
        # Variable dropout.
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), 0, 0.9, step=0.1, default=0.5)))

    model.add(tf.keras.layers.Dense(7, activation='softmax'))  # Output layer

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning rate', [1e-1, 1e-2, 1e-3, 1e-4])),  # Variable learning rate.
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def fs_tuning(fs_alg, x_train, y_train, x_val, y_val, save_path_txt, save_path_model):
    """ Run the hyperparameter tuner for each feature subset. Call with the
    feature selection algorithm you wish to test, e.g. RFE or FSFS. """
    for i in range(5, 40, 5):
        x_train_selec, x_val_selec, feats_selec = fs_alg(x_train, y_train, x_val, i)

        tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=50, hyperband_iterations=3,
                             directory="DIR/TO/OUTPUT/TUNER/RESULTS", project_name="PROJNAME")
        tuner.search(x_train_selec, y_train, validation_data=(x_val_selec, y_val), epochs=50,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

        best_model = tuner.get_best_models(1)[0]

        with open(f"{save_path_txt}/run_result{i}.txt", 'w+') as f:
            f.write("Train accuracy: " + str(best_model.evaluate(x_train_selec, y_train)[1]) + "\n")
            f.write("\nVal accuracy: " + str(best_model.evaluate(x_val_selec, y_val)[1]) + "\n")
            f.write(feature_selection.print_selected_features(data.feature_names, feats_selec) + "\n")
            f.write(str(best_model.get_config()))

        best_model.save(f"{save_path_model}{i}")
