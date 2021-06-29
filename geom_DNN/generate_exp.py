# ==============================================================================
# Create visual explanations for the DNN classifier using geometric features.
# Each of the 40 geometric features are based on 49 landmarks.

# Each feature has an associated (absolute) SHAP feature importance score.

# The n most important features are plotted on the aligned face imaged
# by plotting the landmarks associated with that feature.

# All landmarks belonging to a certain feature have the same colour.
# The higher the feature's score,
# the bigger the landmarks are plotted on the image.

# Also generated are the textual explanations and the probability distributions.
# ==============================================================================

import preprocess
import data

import numpy as np  # https://numpy.org
import pandas as pd  # https://pandas.pydata.org
import matplotlib.pyplot as plt  # https://matplotlib.org
import matplotlib.colors
import os


def get_coordinates_matrix(i, coordinates):
    """ Convert the (49*2, 1) vector to a (49, 2) matrix.
    i is the index where the vector in coordinates is located. """
    vector = coordinates.iloc[i].to_numpy()
    matrix = np.reshape(vector, (2, 49))

    return matrix


def get_shap_features(i, predictions, shap_values, feature_names, n):
    """ Return the n features with the highest absolute SHAP value. """
    vals = shap_values[predictions[i]][i]
    vals = np.abs(vals)

    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature_name', 'importance'])
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)

    total_weight = sum(feature_importance['importance'])
    most_important_weight = sum(feature_importance.head(n)['importance'])
    prop_weight = round(most_important_weight / total_weight * 100, 1)

    return (feature_importance.head(n), prop_weight)


def get_landmarks(features):
    """ Return the landmarks belonging to the given features. """
    landmarks = []
    for feature in features:
        landmarks.extend(data.feature_to_landmark[feature])

    return landmarks


def plot_important_features(plot, image_file, features, coordinates):
    """ Plot the given features on the given image. """
    im = plt.imread(image_file)
    plot.imshow(im, cmap='gray')  # Show the image.

    cmap = plt.cm.YlOrRd  # Colour map is in reds.
    # The colour mapping goes from the minimum shap value to the maximum shap value.
    minShap = min(features['importance'])
    maxShap = max(features['importance'])
    norm = matplotlib.colors.Normalize(vmin=minShap, vmax=maxShap)

    # Display each feature.
    for _, feature in features.iterrows():
        color = cmap(norm(feature['importance']))  # Determine colour for this feature.
        landmarks = data.feature_to_landmark[feature['feature_name']]  # Get corresponding landmarks.
        # Plot the feature.
        data.feature_to_function[feature['feature_name']](plot, coordinates, landmarks, color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plot.axis('off')  # No need for x and y axes.
    plot.colorbar(sm)  # Show the colour bar.


def text_explanation(label_true, label_pred, features, weight, n):
    """ Generate a textual explanation based on the given features.
    The explanation mentions the predicted label and whether the prediction is
    correct. If it is not, the text mentions which label is the right one. """
    label_true_text = data.num_to_label[label_true]
    label_pred_text = data.num_to_label[label_pred]

    if label_true == label_pred:  # A correct classification.
        outcome = "CORRECT."
    else:  # An incorrect one.
        outcome = f"INCORRECT; the correct label is {label_true_text}."

    s = f"This person's emotion is classified as {label_pred_text}. "
    s += f"This classification is {outcome}\n\n"
    s += f"The following {n} features, listed from most important to less important, contributed for {weight}% to the \
    decision: \n\n"

    # List the top n features.
    i = 1
    for feature in features['feature_name']:
        s += f"{i}.\t{data.feature_to_text[feature]}\n"
        i += 1

    return s


def plot_probability_distribution(plot, pred, label_true):
    """ Plot the probability distribution as given in pred in a bar plot. """
    barlist = plot.bar(data.class_names, pred, color='paleturquoise')
    barlist[label_true].set_color('teal')  # Give the bar of the correct class a different colour.

    plot.yticks(np.arange(0, 1.1, 0.1))
    # Legend
    handle = [plt.Rectangle((0, 0), 1, 1, color='teal'), plt.Rectangle((0, 0), 1, 1, color='paleturquoise')]
    plt.legend(handle, ['True label', 'Other label'])

    plot.title('Probability distribution')


def generate_all_exp(filenames, prob_pred, shap_values, feature_set, labels, coordinates, save_path, n):
    """ Generate explanations for all files. """
    i = 0  # Index
    # Get the class with the highest probability (i.e. the class the model predicts).
    pred = np.apply_along_axis(np.argmax, 1, prob_pred)

    for file in filenames:
        image = os.path.join(TEST_IMAGE_FOLDER, file)

        # Where to save the explanation image.
        save_path_exp = os.path.join(save_path, f"{i}.png")
        # Where to save probability distribution plots.
        save_path_pd = os.path.join(save_path, f"{i}_pd.png")
        # Where to save text explanations.
        save_path_txt = os.path.join(save_path, f"{i}.txt")

        # Get important features for the image.
        features, prop_weight = get_shap_features(i, pred, shap_values, feature_set, n)

        # Plot the features.
        coords = get_coordinates_matrix(i, coordinates)
        plot_important_features(plt, image, features, coords)

        plt.savefig(save_path_exp, bbox_inches='tight', dpi=600)  # Save the image.
        plt.clf()  # Clear the plot for the next one.

        # Plot class probability distribution.
        plot_probability_distribution(plt, prob_pred[i], labels.iloc[i])

        plt.savefig(save_path_pd, bbox_inches='tight', dpi=600)
        plt.clf()  # Clear the plot.

        # Generate textual explanation and save to file.
        s = text_explanation(labels.iloc[i], pred[i], features, prop_weight, n)

        with open(save_path_txt, 'w+') as f:
            f.write(s)

        i += 1


def plot_landmarks(coords, filename, savefile=None):
    """ Simply plot the landmarks on their correct position on the face. 
    Each landmark is annotated with its index. """
    im = plt.imread("test_3pose_aligned/" + filename)
    plt.imshow(im, cmap='gray')

    plt.scatter(coords[0], coords[1], s=10, c='red')
    plt.axis('off')

    for i in range(0, len(coords[0])):
        plt.annotate(i, xy=(coords[0][i], coords[1][i]), color='white')

    if savefile:
        plt.savefig(savefile, dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    TEST_IMAGE_FOLDER = "PATH/TO/TEST_SET_IMAGES"

    # Load the landmark coordinates data sets splitted on pose.
    file_test = "PATH/TO/LANDMARKS.CSV"
    df_S, df_HL, df_HR = preprocess.load_data(file_test, True)

    # Get predicted labels for each test set.
    predS = data.modelS.predict(data.x_test_standS)
    predHL = data.modelHL.predict(data.x_test_standHL)
    predHR = data.modelHR.predict(data.x_test_standHR)

    # Get coordinates for each image.
    coords_S, _ = preprocess.get_x_and_y(df_S)
    coords_HL, _ = preprocess.get_x_and_y(df_HL)
    coords_HR, _ = preprocess.get_x_and_y(df_HR)

    # Split the image files on pose.
    filenames_S = []
    filenames_HL = []
    filenames_HR = []
    for filename in sorted(os.listdir(TEST_IMAGE_FOLDER)):
        if filename.endswith("S.JPG"):
            filenames_S.append(filename)
        elif filename.endswith("HL.JPG"):
            filenames_HL.append(filename)
        else:
            filenames_HR.append(filename)

    n = 5  # Number of features to display.

    # Generate explanations for all test images!
    generate_all_exp(filenames_S, predS, data.shap_modelS, data.feature_set_S, 
                     data.y_testS, coords_S, "PATH/TO/SAVE/EXPLANATIONS/S", n)
    generate_all_exp(filenames_HL, predHL, data.shap_modelHL, data.feature_set_HL, 
                     data.y_testHL, coords_HL, "PATH/TO/SAVE/EXPLANATIONS/HL", n)
    generate_all_exp(filenames_HR, predHR, data.shap_modelHR, data.feature_set_HR, 
                     data.y_testHR, coords_HR, "PATH/TO/SAVE/EXPLANATIONS/HR", n)
