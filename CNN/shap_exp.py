# ==============================================================================
# Generate SHAP explanations for images using the GradientExplainer.
# Note that explanations for the CNN generated from SHAP are not used in the
# final paper.
# ==============================================================================

import load_data

import numpy as np  # https://numpy.org
import tensorflow as tf  # https://www.tensorflow.org
import shap  # https://github.com/slundberg/shap
import matplotlib.pyplot as plt  # https://matplotlib.org

tf.compat.v1.disable_eager_execution()  # Otherwise shap doesn't work.

model = tf.keras.models.load_model("PATH/TO/MODEL")

x, labels = load_data.load_data_np('aligned_images/val')

class_names = {0: 'AN', 1: 'DI', 2: 'FE', 3: 'HA', 4: 'NE', 5: 'SA', 6: 'SU'}

to_explain = x[[40]]  # Instance for which to generate the explanation.

e = shap.GradientExplainer(model, x, local_smoothing=100)
shap_values, indexes = e.shap_values(to_explain, ranked_outputs=1)

index_names = np.vectorize(lambda l: class_names[l])(indexes)

shap.image_plot(shap_values, to_explain, index_names, show=False)
plt.show()
