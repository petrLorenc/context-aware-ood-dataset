from AbstractNeuralNet import AbstractNeuralNet
from utils import EXTRA_LAYER_ACT_F
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras import layers, activations


class OwnLogisticRegression(AbstractNeuralNet):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(num_classes, activation=activations.sigmoid),
            layers.Activation(activations.softmax)])
        return model

    def threshold(self, val_predictions_labels, oos_label, value=0.5):
        """
            for pred, true_label in val_predictions_labels:
                pred_label = pred[0]
                similarity = pred[1]
        """
        return None


class BaselineNNExtraLayer(AbstractNeuralNet):
    """Baseline Neural Net Extra Layer"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(emb_dim, activation=EXTRA_LAYER_ACT_F),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation=activations.softmax)])

        return model

    def threshold(self, val_predictions_labels, oos_label, value=0.5):
        """
            for pred, true_label in val_predictions_labels:
                pred_label = pred[0]
                similarity = pred[1]
        """
        return None
