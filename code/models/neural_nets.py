import numpy as np
from abc import ABCMeta

import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.callbacks import EarlyStopping

from models import AbstractModel


class AbstractNeuralNet(AbstractModel, metaclass=ABCMeta):
    def __init__(self, loss_function=None, epochs=None, batch_size=None, init_learning_rate=None):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.loss_function = loss_function
        self.early_stopping_patience = 10

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "init_learning_rate": self.init_learning_rate,
            "loss_function": str(self.loss_function)
        }

    def fit(self, X_train, y_train, X_val, y_val):
        emb_dim = X_train.shape[1]  # embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes

        self.model = self.create_model(emb_dim, num_classes)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.init_learning_rate),
                           loss=self.loss_function,
                           metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', mode="min", patience=self.early_stopping_patience)

        try:
            self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                           callbacks=[es])
        except KeyboardInterrupt:
            print("Training stopped by CTRL+C")

        return self.model

    def predict(self, X_test):
        """Returns predictions with class labels."""

        probs = self.model.predict(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions

    def predict_proba(self, X_test):
        """Returns probabilities of each label."""

        probs = self.model.predict(X_test)

        return probs

    def predict_with(self, model, X_test):
        """Returns predictions with class labels."""

        probs = model.predict(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions

    def predict_proba_with(self, model, X_test):
        """Returns probabilities of each label."""

        probs = model.predict(X_test)

        return probs

class OwnLogisticRegression(AbstractNeuralNet):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(num_classes, activation=activations.sigmoid),
            layers.Activation(activations.softmax)])
        return model


class BaselineNNExtraLayer(AbstractNeuralNet):
    """Baseline Neural Net Extra Layer"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(emb_dim, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation=activations.softmax)])

        return model

