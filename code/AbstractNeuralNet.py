from utils import EXTRA_LAYER_ACT_F

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import losses, layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class AbstractNeuralNet(ABC):
    def __init__(self, loss=losses.SparseCategoricalCrossentropy()):
        tf.random.set_seed(7)  # set seed in order to have reproducible results

        self.model = None
        self.model_name = type(self).__name__
        self.loss = loss
        self.oos_label = None  # used for CosFaceLofNN

    @abstractmethod
    def create_model(self, emb_dim, num_classes):
        raise NotImplementedError('You have to create a model.')

    def fit(self, X_train, y_train, X_val, y_val):
        emb_dim = X_train.shape[1]  # embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes

        self.model = self.create_model(emb_dim, num_classes)

        self.model.compile(optimizer='adam',
                           loss=self.loss,
                           metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', mode="min", patience=5)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=250, verbose=0, callbacks=[es])


    def predict(self, X_test):
        """Returns predictions with class labels."""

        probs = self.model.predict(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions

    def predict_proba(self, X_test):
        """Returns probabilities of each label."""

        probs = self.model.predict(X_test)

        return probs