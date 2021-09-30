from AbstractNeuralNet import AbstractNeuralNet
from utils import EXTRA_LAYER_ACT_F

import tensorflow as tf
from tensorflow.keras import layers, activations


class BaselineNN(AbstractNeuralNet):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(num_classes, activation=activations.softmax)])

        return model


class BaselineNNExtraLayer(AbstractNeuralNet):
    """Baseline Neural Net Extra Layer"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(emb_dim, activation=EXTRA_LAYER_ACT_F),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation=activations.softmax)])

        return model