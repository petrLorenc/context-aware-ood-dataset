from AbstractNeuralNet import AbstractSklearnModel
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, activations


class SklearnLogisticRegression(AbstractSklearnModel):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = LogisticRegression(max_iter=850,
                                   penalty='l2',
                                   random_state=42,
                                   solver="lbfgs",
                                   class_weight="balanced",
                                   dual=False,
                                   multi_class="multinomial")
        return model

    def threshold(self, val_predictions_labels, oos_label, value=0.5):
        """
            for pred, true_label in val_predictions_labels:
                pred_label = pred[0]
                similarity = pred[1]
        """
        return None
