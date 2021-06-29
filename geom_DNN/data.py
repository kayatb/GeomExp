# ==============================================================================
# Contains data used by the other scripts:
# - feature sets/names
# - class names
# - numbers to class names mapping
# - train, validation and test sets (split on pose)
# - filenames
# - models
# - shap values per model
# - feature name to landmark numbers mapping
# - feature name to plot function mapping
# - feature name to feature description mapping

# All path names mentioned here are placeholders. 
# ==============================================================================

import preprocess
import plot_features as pf

import tensorflow as tf  # https://www.tensorflow.org
import numpy as np  # https://numpy.org

# ==============================================================================
# ============= FEATURE SETS ============
# ==============================================================================
feature_set_S = [
    "Eye aspect ratio (L)", "Mouth aspect ratio", "Upper lip angle (L)", "Nose tip - mouth corner angles (L)",
    "Nose tip - mouth corner angles (R)", "Lower lip angle (L)", "Lower lip angles (R)", "Eyebrow slope (L)",
    "Lower eye outer angles (L)", "Lower eye inner angles (L)", "Lower eye outer angles (R)",
    "Mouth corner - mouth bottom angle (L)", "Mouth corner - mouth bottom angle (R)", "Upper mouth angles (L)",
    "Upper mouth angles (R)", "Curvature of lower-outer lips (R)", "Curvature of lower-inner lips (L)",
    "Curvature of lower-inner lips (R)", "Mouth opening / mouth width", "Mouth up/low",
    "Eye - middle eyebrow distance (L)", "Eye - middle eyebrow distance (R)", "Eye - inner eyebrow distance (L)",
    "Eye - inner eyebrow distance (R)", "Inner eye - eyebrow center (L)", "Inner eye - eyebrow center (R)",
    "Inner eye - mouth top distance (R)", "Mouth width", "Outer mid eyebrow slope (L)", "Outer mid eyebrow slope (R)"
]

feature_set_HL = [
    "Eye aspect ratio (L)", "Mouth aspect ratio", "Upper lip angle (L)", "Upper lip angle (R)",
    "Nose tip - mouth corner angles (L)", "Nose tip - mouth corner angles (R)", "Lower lip angles (R)",
    "Lower eye outer angles (R)", "Mouth corner - mouth bottom angle (L)", "Upper mouth angles (L)",
    "Upper mouth angles (R)", "Curvature of lower-inner lips (R)", "Mouth opening / mouth width", "Mouth up/low",
    "Eye - middle eyebrow distance (L)", "Eye - middle eyebrow distance (R)", "Eye - inner eyebrow distance (L)",
    "Eye - inner eyebrow distance (R)", "Inner eye - eyebrow center (L)", "Inner eye - mouth top distance (L)",
    "Inner eye - mouth top distance (R)", "Mouth height", "Upper mouth height", "Outer mid eyebrow slope (L)",
    "Outer mid eyebrow slope (R)"
]

feature_set_HR = [
    "Eye aspect ratio (L)", "Mouth aspect ratio", "Upper lip angle (L)", "Nose tip - mouth corner angles (L)",
    "Lower lip angle (L)", "Lower lip angles (R)", "Eyebrow slope (L)", "Eyebrow slope (R)",
    "Lower eye outer angles (L)", "Lower eye inner angles (L)", "Lower eye outer angles (R)",
    "Lower eye inner angles (R)", "Mouth corner - mouth bottom angle (L)", "Mouth corner - mouth bottom angle (R)",
    "Upper mouth angles (L)", "Curvature of lower-outer lips (L)", "Curvature of lower-outer lips (R)",
    "Curvature of lower-inner lips (L)", "Curvature of lower-inner lips (R)", "Mouth opening / mouth width",
    "Mouth up/low", "Eye - inner eyebrow distance (L)", "Eye - inner eyebrow distance (R)",
    "Inner eye - eyebrow center (L)", "Inner eye - eyebrow center (R)", "Inner eye - mouth top distance (L)",
    "Mouth width", "Mouth height", "Outer mid eyebrow slope (L)", "Outer mid eyebrow slope (R)"
]

# ==============================================================================
# ============= CLASS NAMES =============
# ==============================================================================
class_names = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA', 'SU']

# Numerical representation used for the features during classification.
num_to_label = {0: 'AFRAID', 1: 'ANGER', 2: 'DISGUST', 3: 'HAPPY', 4: 'NEUTRAL', 5: 'SAD', 6: 'SURPRISE'}

# ==============================================================================
# ============= DATA SETS =============
# ==============================================================================
file_train = "PATH/TO/TRAINING_SET"
file_val = "PATH/TO/VALIDATION_SET"
file_test = "PATH/TO/TEST_SET"

# Load datasets without splitting on pose.
# df_train = preprocess.load_data(file_train, False)
# df_val = preprocess.load_data(file_val, False)
# x_train, y_train = preprocess.get_x_and_y(df_train)
# x_val, y_val = preprocess.get_x_and_y(df_val)
# x_train, x_val = preprocess.standardize_data(x_train, x_val)

# Load datasets splitted on pose.
dfS_train, dfHL_train, dfHR_train = preprocess.load_data(file_train, True)
dfS_val, dfHL_val, dfHR_val = preprocess.load_data(file_val, True)
dfS_test, dfHL_test, dfHR_test = preprocess.load_data(file_test, True)

# Filename of each datapoint.
file_namesS = dfS_val["Filename"]
file_namesHL = dfHL_val["Filename"]
file_namesHR = dfHR_val["Filename"]

# Only keep the features that were selected during feature selection.
dfS_train = dfS_train[["Filename"] + feature_set_S]
dfS_val = dfS_val[["Filename"] + feature_set_S]
dfS_test = dfS_test[["Filename"] + feature_set_S]

dfHL_train = dfHL_train[["Filename"] + feature_set_HL]
dfHL_val = dfHL_val[["Filename"] + feature_set_HL]
dfHL_test = dfHL_test[["Filename"] + feature_set_HL]

dfHR_train = dfHR_train[["Filename"] + feature_set_HR]
dfHR_val = dfHR_val[["Filename"] + feature_set_HR]
dfHR_test = dfHR_test[["Filename"] + feature_set_HR]

# Split each set in x and y. Standardize the x.
# S set
x_trainS, y_trainS = preprocess.get_x_and_y(dfS_train)
x_valS, y_valS = preprocess.get_x_and_y(dfS_val)
x_testS, y_testS = preprocess.get_x_and_y(dfS_test)

x_train_standS, x_val_standS = preprocess.standardize_data(x_trainS, x_valS)
_, x_test_standS = preprocess.standardize_data(x_trainS, x_testS)

# HL set
x_trainHL, y_trainHL = preprocess.get_x_and_y(dfHL_train)
x_valHL, y_valHL = preprocess.get_x_and_y(dfHL_val)
x_testHL, y_testHL = preprocess.get_x_and_y(dfHL_test)

x_train_standHL, x_val_standHL = preprocess.standardize_data(x_trainHL, x_valHL)
_, x_test_standHL = preprocess.standardize_data(x_trainHL, x_testHL)

# HR set
x_trainHR, y_trainHR = preprocess.get_x_and_y(dfHR_train)
x_valHR, y_valHR = preprocess.get_x_and_y(dfHR_val)
x_testHR, y_testHR = preprocess.get_x_and_y(dfHR_test)

x_train_standHR, x_val_standHR = preprocess.standardize_data(x_trainHR, x_valHR)
_, x_test_standHR = preprocess.standardize_data(x_trainHR, x_testHR)

# Names of all features.
dfS_train.drop("Filename", axis=1)
feature_names = dfS_train.columns

# ==============================================================================
# ============= MODELS =============
# ==============================================================================
modelS = tf.keras.models.load_model("PATH/TO/FINAL/MODEL/S-SET")
modelHL = tf.keras.models.load_model("PATH/TO/FINAL/MODEL/HL-SET")
modelHR = tf.keras.models.load_model("PATH/TO/FINAL/MODEL/HR-SET")


# ==============================================================================
# ============= SHAP VALUES =============
# ==============================================================================
shap_modelS = {
    0: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/AF.CSV", delimiter=','),
    1: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/AN.CSV", delimiter=','),
    2: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/DI.CSV", delimiter=','),
    3: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/HA.CSV", delimiter=','),
    4: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/NE.CSV", delimiter=','),
    5: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/SA.CSV", delimiter=','),
    6: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-S/SU.CSV", delimiter=',')
}

shap_modelHL = {
    0: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=','),
    1: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=','),
    2: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=','),
    3: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=','),
    4: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=','),
    5: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=','),
    6: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HL/AF.CSV", delimiter=',')
}

shap_modelHR = {
    0: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=','),
    1: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=','),
    2: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=','),
    3: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=','),
    4: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=','),
    5: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=','),
    6: np.genfromtxt("PATH/TO/SHAP-VALUES/MODEL-HR/AF.CSV", delimiter=',')
}


# ==============================================================================
# ============= LANDMARK TO GEOMETRIC FEATURES MAPPING =============
# ==============================================================================
# Which landmarks belong to which feature. Used for plotting.
feature_to_landmark = {  # The landmarks should remain in this order for plotting!
    'Eye aspect ratio (L)': [21, 20, 23, 24, 19, 22],
    'Eye aspect ratio (R)': [26, 27, 30, 29, 25, 28],
    'Mouth aspect ratio': [31, 37, 34, 40],
    'Upper lip angle (L)': [31, 34],
    'Upper lip angle (R)': [37, 34],
    'Nose tip - mouth corner angles (L)': [31, 16],
    'Nose tip - mouth corner angles (R)': [37, 16],
    'Lower lip angle (L)': [41, 31],
    'Lower lip angles (R)': [39, 37],
    'Eyebrow slope (L)': [0, 4],
    'Eyebrow slope (R)': [9, 5],
    'Lower eye outer angles (L)': [19, 24],
    'Lower eye inner angles (L)': [22, 23],
    'Lower eye outer angles (R)': [28, 29],
    'Lower eye inner angles (R)': [25, 30],
    'Mouth corner - mouth bottom angle (L)': [31, 40],
    'Mouth corner - mouth bottom angle (R)': [37, 40],
    'Upper mouth angles (L)': [33, 40],
    'Upper mouth angles (R)': [35, 40],
    'Curvature of lower-outer lips (L)': [41, 42, 31],
    'Curvature of lower-outer lips (R)': [37, 38, 39],
    'Curvature of lower-inner lips (L)': [31, 41, 40],
    'Curvature of lower-inner lips (R)': [37, 39, 40],
    'Bottom lip curvature': [37, 40, 31],
    'Mouth opening / mouth width': [44, 47, 43, 48, 45, 46],
    'Mouth up/low': [40, 44, 34],
    'Eye - middle eyebrow distance (L)': [0, 4, 19, 22],
    'Eye - middle eyebrow distance (R)': [9, 5, 28, 25],
    'Eye - inner eyebrow distance (L)': [19, 22, 4],
    'Eye - inner eyebrow distance (R)': [28, 25, 5],
    'Inner eye - eyebrow center (L)': [22, 2],
    'Inner eye - eyebrow center (R)': [25, 7],
    'Inner eye - mouth top distance (L)': [22, 34],
    'Inner eye - mouth top distance (R)': [25, 34],
    'Mouth width': [31, 37],
    'Mouth height': [34, 40],
    'Upper mouth height': [44, 47, 34],
    'Lower mouth height': [44, 47, 40],
    'Outer mid eyebrow slope (L)': [0, 2],
    'Outer mid eyebrow slope (R)': [9, 7]
}

# ==============================================================================
# ============= FEATURE PLOTTING FUNCTIONS =============
# ==============================================================================
# Dictionary mapping feature names to the corresponding plotting functions.
feature_to_function = {
    'Eye aspect ratio (L)': pf.plot_ratio_six_points,
    'Eye aspect ratio (R)': pf.plot_ratio_six_points,
    'Mouth aspect ratio': pf.plot_ratio_four_points,
    'Upper lip angle (L)': pf.plot_angle_horizontal,
    'Upper lip angle (R)': pf.plot_angle_horizontal,
    'Nose tip - mouth corner angles (L)': pf.plot_angle_vertical,
    'Nose tip - mouth corner angles (R)': pf.plot_angle_vertical,
    'Lower lip angle (L)': pf.plot_angle_horizontal,
    'Lower lip angles (R)': pf.plot_angle_horizontal,
    'Eyebrow slope (L)': pf.plot_angle_horizontal,
    'Eyebrow slope (R)': pf.plot_angle_horizontal,
    'Lower eye outer angles (L)': pf.plot_angle_horizontal,
    'Lower eye inner angles (L)': pf.plot_angle_horizontal,
    'Lower eye outer angles (R)': pf.plot_angle_horizontal,
    'Lower eye inner angles (R)': pf.plot_angle_horizontal,
    'Mouth corner - mouth bottom angle (L)': pf.plot_angle_horizontal,
    'Mouth corner - mouth bottom angle (R)': pf.plot_angle_horizontal,
    'Upper mouth angles (L)': pf.plot_angle_horizontal,
    'Upper mouth angles (R)': pf.plot_angle_horizontal,
    'Curvature of lower-outer lips (L)': pf.plot_curve,
    'Curvature of lower-outer lips (R)': pf.plot_curve,
    'Curvature of lower-inner lips (L)': pf.plot_curve,
    'Curvature of lower-inner lips (R)': pf.plot_curve,
    'Bottom lip curvature': pf.plot_curve,
    'Mouth opening / mouth width': pf.plot_ellipse,
    'Mouth up/low': pf.plot_three_points_line,
    'Eye - middle eyebrow distance (L)': pf.plot_line_two_centres,
    'Eye - middle eyebrow distance (R)': pf.plot_line_two_centres,
    'Eye - inner eyebrow distance (L)': pf.plot_three_points_centre,
    'Eye - inner eyebrow distance (R)': pf.plot_three_points_centre,
    'Inner eye - eyebrow center (L)': pf.plot_two_points_line,
    'Inner eye - eyebrow center (R)': pf.plot_two_points_line,
    'Inner eye - mouth top distance (L)': pf.plot_two_points_line,
    'Inner eye - mouth top distance (R)': pf.plot_two_points_line,
    'Mouth width': pf.plot_two_points_line,
    'Mouth height': pf.plot_two_points_line,
    'Upper mouth height': pf.plot_three_points_centre,
    'Lower mouth height': pf.plot_three_points_centre,
    'Outer mid eyebrow slope (L)': pf.plot_two_points_line,
    'Outer mid eyebrow slope (R)': pf.plot_two_points_line
}


# ==============================================================================
# ============= TEXTUAL FEATURE DESCRIPTIONS =============
# ==============================================================================
# More informative feature descriptions. Used for the textual explanations.

feature_to_text = {
    'Eye aspect ratio (L)': "Left eye aspect ratio (ratio between eye width and eye height)",
    'Eye aspect ratio (R)': "Right eye aspect ratio (ratio between eye width and eye height)",
    'Mouth aspect ratio': "Mouth aspect ratio (ratio between mouth width and mouth height)",
    'Upper lip angle (L)': "Angle from left mouth corner to top of the mouth",
    'Upper lip angle (R)': "Angle from right mouth corner to top of the mouth",
    'Nose tip - mouth corner angles (L)': "Angle from nose tip to left mouth corner",
    'Nose tip - mouth corner angles (R)': "Angle from nose tip to right mouth corner",
    'Lower lip angle (L)': "Angle from left lower lip to left mouth corner",
    'Lower lip angles (R)': "Angle from right lower lip to right mouth corner",
    'Eyebrow slope (L)': "Left eyebrow angle",
    'Eyebrow slope (R)': "Right eyebrow angle",
    'Lower eye outer angles (L)': "Left lower eye outer angle",
    'Lower eye inner angles (L)': "Left lower eye inner angle",
    'Lower eye outer angles (R)': "Right lower eye outer angle",
    'Lower eye inner angles (R)': "Right lower eye inner angle",
    'Mouth corner - mouth bottom angle (L)': "Angle from left mouth corner to mouth bottom",
    'Mouth corner - mouth bottom angle (R)': "Angle from right mouth corner to mouth bottom",
    'Upper mouth angles (L)': "Angle from left mouth corner to left upper mouth",
    'Upper mouth angles (R)': "Angle from right mouth corner to right upper mouth",
    'Curvature of lower-outer lips (L)': "Curve of the left lower-outer lip",
    'Curvature of lower-outer lips (R)': "Curve of the right lower-outer lip",
    'Curvature of lower-inner lips (L)': "Curve of the left lower-inner lip",
    'Curvature of lower-inner lips (R)': "Curve of the right lower-inner lip",
    'Bottom lip curvature': "Curve of the bottom lip",
    'Mouth opening / mouth width': "Opening of the mouth",
    'Mouth up/low': "Ratio between the lower mouth height and the upper mouth height",
    'Eye - middle eyebrow distance (L)': "Distance between the centre of the left eye \
                                          and the centre of the left eyebrow",
    'Eye - middle eyebrow distance (R)': "Distance between the centre of the right eye \
                                          and the centre of the right eyebrow",
    'Eye - inner eyebrow distance (L)': "Distance between the centre of the left eye and the left inner eyebrow",
    'Eye - inner eyebrow distance (R)': "Distance between the centre of the right eye and the right inner eyebrow",
    'Inner eye - eyebrow center (L)': "Distance between the left inner eye and the centre of the left eyebrow",
    'Inner eye - eyebrow center (R)': "Distance between the right inner eye and the centre of the right eyebrow",
    'Inner eye - mouth top distance (L)': "Distance between the left inner eye and the mouth top",
    'Inner eye - mouth top distance (R)': "Distance between the right inner eye and the mouth top",
    'Mouth width': "Width of the mouth",
    'Mouth height': "Height of the mouth",
    'Upper mouth height': "Distance from the mouth centre to the top of the mouth",
    'Lower mouth height': "Distance from the mouth centre to the bottom of the mouth",
    'Outer mid eyebrow slope (L)': "Slope of the outer to centre part of the left eyebrow",
    'Outer mid eyebrow slope (R)': "Slope of the outer to centre part of the right eyebrow"
}
