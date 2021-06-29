# ==============================================================================
# Several functions for performing feature selection.
# Implemented algorithms:
# - Forward Sequential Feature Selection (FSFS)
# - Recursive Feature Elimination (RFE)
# - SHAP feature selection
# ==============================================================================

import pandas as pd  # https://pandas.pydata.org
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, RFE  # https://scikit-learn.org/stable/
from sklearn.linear_model import LogisticRegression


def sequential_feature_selection(x_train, y_train, x_test, n, direction='forward'):
    """ Choose a subset of n features in a greedy fashion.
    Can be done forwards or backwards.
    Fits a logistic regression model as an estimator. """
    log = LogisticRegression()
    sfs = SequentialFeatureSelector(log, n_features_to_select=n, direction=direction)
    sfs.fit(x_train, y_train)

    x_train_selected = sfs.transform(x_train)
    x_test_selected = sfs.transform(x_test)
    # Return an array of booleans with whether or not the feature was selected.
    return x_train_selected, x_test_selected, sfs.get_support()


def recursive_feature_elimination(x_train, y_train, x_test, n):
    """ Recursively consider a smaller and smaller subset of features.
    Fits a logistic regression model as an estimator. """
    log = LogisticRegression()
    selector = RFE(log, n_features_to_select=n)
    selector.fit(x_train, y_train)

    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)
    # Return an array of booleans with whether or not the feature was selected.
    return x_train_selected, x_test_selected, selector.get_support()


def shap_feature_selection(x_train, y_train, x_test, n):
    """ Select the n-best features based on the calculated SHAP importance. """
    fs = SelectKBest(score_func=get_shap_score, k=n)
    fs.fit(x_train, y_train)
    x_train_selected = fs.transform(x_train)
    x_test_selected = fs.transform(x_test)

    return x_train_selected, x_test_selected


def get_shap_score(x, y):
    """ Helper function for shap feature selection. """
    shap_importance = pd.read_csv("shap_values/no_selection/modelHR/feature_importance.csv", header=0)
    return shap_importance['importance']


def print_selected_features(column_names, selection):
    """ Print the names of the features that are kept after selection. """
    s = ""
    for col, sel in zip(column_names, selection):
        if sel:
            s += f"{col}, "
    return s
