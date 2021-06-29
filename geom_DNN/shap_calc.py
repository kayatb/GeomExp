# ==============================================================================
# Function for calculating the SHAP values for each datapoint and the global
# SHAP values. Also implemented are function for calculating fidelity and
# explanation accuracy metrics.
# ==============================================================================

import data

import os
import numpy as np  # https://numpy.org
import pandas as pd  # https://pandas.pydata.org
import shap  # https://github.com/slundberg/shap
import tensorflow as tf  # https://www.tensorflow.org
import matplotlib.pyplot as plt  # https://matplotlib.org

# Otherwise shap doesn't work -> AttributeError: 'KerasTensor' object has no attribute 'graph'
tf.compat.v1.disable_eager_execution()
# ATTENTION: the above line means the models loaded in data.py cannot be used.
# load the models here separately.

SHAP_PATH = "PATH/TO/SAVE/SHAP-VALUES"


def save_shap_values(shap_values, model_name):
    """ Save the SHAP values to a csv-file per label. """
    np.savetxt(f"{SHAP_PATH}/{model_name}/AF.csv", shap_values[0], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "AF.csv"), shap_values[0], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "AN.csv"), shap_values[1], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "DI.csv"), shap_values[2], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "HA.csv"), shap_values[3], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "NE.csv"), shap_values[4], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "SA.csv"), shap_values[5], delimiter=",", fmt="%.3e")
    np.savetxt(os.path.join(SHAP_PATH, model_name, "SU.csv"), shap_values[6], delimiter=",", fmt="%.3e")


def calc_shap_values(model, x_train, x):
    """ Calculate the SHAP values for x. Background is always the train set. """
    background = x_train
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x)

    return shap_values


def calc_feature_importance(shap_values, save_file=None):
    """ Calculate the total feature importance for each feature according to the
    SHAP values. This is calculated by taking the absolute SHAP values for each feature
    for each label. Then average the SHAP value per feature for each label. Finally, summate
    the average values per labels in one total SHAP value per feature.
    shap_values should be a dict containing a numpy array per label (as in data.py). """

    # Take absolute SHAP values.
    shap_AF = np.abs(shap_values[0], delimiter=',')
    shap_AN = np.abs(shap_values[1], delimiter=',')
    shap_DI = np.abs(shap_values[2], delimiter=',')
    shap_HA = np.abs(shap_values[3], delimiter=',')
    shap_NE = np.abs(shap_values[4], delimiter=',')
    shap_SA = np.abs(shap_values[5], delimiter=',')
    shap_SU = np.abs(shap_values[6], delimiter=',')

    # Take the average SHAP value per feature.
    shap_AF_avg = np.mean(shap_AF, axis=0)
    shap_AN_avg = np.mean(shap_AN, axis=0)
    shap_DI_avg = np.mean(shap_DI, axis=0)
    shap_HA_avg = np.mean(shap_HA, axis=0)
    shap_NE_avg = np.mean(shap_NE, axis=0)
    shap_SA_avg = np.mean(shap_SA, axis=0)
    shap_SU_avg = np.mean(shap_SU, axis=0)

    # Calculate the total SHAP value by summing over all classes.
    total_shap_value = np.add(shap_AF_avg, shap_AN_avg)
    total_shap_value = np.add(total_shap_value, shap_DI_avg)
    total_shap_value = np.add(total_shap_value, shap_HA_avg)
    total_shap_value = np.add(total_shap_value, shap_NE_avg)
    total_shap_value = np.add(total_shap_value, shap_SA_avg)
    total_shap_value = np.add(total_shap_value, shap_SU_avg)

    feature_importance = pd.DataFrame(list(zip(data.feature_set_S, total_shap_value)),
                                      columns=['feature_name', 'importance'])
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)

    if save_file:
        feature_importance.to_csv(save_file)

    return feature_importance


def plot_feature_importance(feature_importance, title, save_file=None):
    """ Plot the feature importances in a bar plot. """
    plt.figure()
    plt.bar([i for i in range(len(feature_importance['importance']))], feature_importance['importance'],
            color="lightskyblue")
    plt.title(title, size=16)
    plt.xlabel("Features", size=12, color="black")
    plt.ylabel("Importance Score", size=12, color="black")

    if save_file:
        plt.savefig(save_file, dpi=600)

    plt.show()


def get_important_features(model, x, index, feature_set, shap_values):
    """ Get the features sorted by importance (i.e. SHAP value) for the
    data point at the given index. """
    pred = np.argmax(model.predict(x[index:index + 1]), axis=-1)
    shap_value = np.abs(shap_values[pred[0]][index])  # Get SHAP values for the predicted class.

    feature_importance = pd.DataFrame(shap_value, index=feature_set, columns=['importance'])
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)

    return feature_importance


def calc_fidelity(model, feature_set, shap_values, x_train, x_test, n):
    """ Calculate the fidelity of the SHAP explanation method. For each data
    instance, use the values of the top n features as is and the mean value for
    all other features. Calculate the model's prediction for such a data point.
    If the prediction stays the same +1 for the fidelity score. """
    score = 0

    # Get average value of each feature.
    feature_avg = x_train.mean(axis=0)  # If we standardise the data, all values will be ~0
    feature_avg = pd.DataFrame(feature_avg, index=feature_set, columns=['avg'])

    # Calculate for each data point.
    for i, val in enumerate(x_test):
        pred_old = model.predict_classes([[val]])  # Prediction for the original data point.
        data_point = pd.DataFrame(val, index=feature_set, columns=['val'])  # The original data point.

        # The n most important features for this data point according to its SHAP values.
        important_features = get_important_features(model, x_test, i, feature_set, shap_values).head(n)

        for feature in feature_set:
            # Check for each feature if it's one of the n most important ones.
            # If not, change the value for this feature to the data set average.
            if feature not in important_features.index:
                data_point.at[feature, 'val'] = feature_avg.at[feature, 'avg']

        data_point = [val[0] for val in data_point.to_numpy()]  # Convert it to a format appropriate for the model.

        pred_new = model.predict_classes([[data_point]])  # Get the prediction for the newly formed data point.

        # The prediction stays the same, so increase the fidelity score.
        if pred_new == pred_old:
            score += 1

    return score / len(x_test)  # Return the score as a percentage of the data set.


def calc_acc(model, feature_set, shap_values, x_train, x_test, y_test, n):
    """ Calculate the fidelity of the SHAP explanation method. For each data
    instance, use the values of the top n features as is and the mean value for
    all other features. Calculate the model's prediction for such a data point.
    If the prediction stays the same +1 for the fidelity score. """
    score = 0

    # Get average value of each feature.
    feature_avg = x_train.mean(axis=0)  # If we standardise the data, all values will be ~0
    feature_avg = pd.DataFrame(feature_avg, index=feature_set, columns=['avg'])

    # Calculate for each data point.
    for i, val in enumerate(x_test):
        # pred_old = model.predict_classes([[val]])  # Prediction for the original data point.
        data_point = pd.DataFrame(val, index=feature_set, columns=['val'])  # The original data point.

        # The n most important features for this data point according to its SHAP values.
        important_features = get_important_features(model, x_test, i, feature_set, shap_values).head(n)

        for feature in feature_set:
            # Check for each feature if it's one of the n most important ones.
            # If not, change the value for this feature to the data set average.
            if feature not in important_features.index:
                data_point.at[feature, 'val'] = feature_avg.at[feature, 'avg']

        data_point = [val[0] for val in data_point.to_numpy()]  # Convert it to a format appropriate for the model.
        # Get the prediction for the newly formed data point.
        pred_new = np.argmax(model.predict([[data_point]]), axis=-1)

        # The prediction stays the same, so increase the fidelity score.
        if pred_new == y_test[i]:
            score += 1

    return score / len(x_test)  # Return the score as a percentage of the data set.


def calc_avg_shap_weights(model, feature_set, shap_values, x, n):
    """ Calculate the shap values of the top n features proportionate to the
    total shap value average over each instance in x. """
    all_prop_weights = 0
    for i, _ in enumerate(x):
        feature_importance = get_important_features(model, x, i, feature_set, shap_values)

        total_weight = sum(feature_importance['importance'])
        most_important_weight = sum(feature_importance.head(n)['importance'])

        prop_weight = most_important_weight / total_weight
        all_prop_weights += prop_weight

    return all_prop_weights / len(x)
