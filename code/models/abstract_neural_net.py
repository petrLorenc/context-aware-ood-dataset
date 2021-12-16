from utils.utils import EXTRA_LAYER_ACT_F

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import losses, layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import sklearn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV


class AbstractModel(ABC):
    def __init__(self):
        np.random.seed(7)  # set seed in order to have reproducible results

        self.model = None
        self.model_name = type(self).__name__

    @staticmethod
    def predict_with(model, X_test):
        """Returns predictions with class labels."""

        probs = model.predict_proba(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions

    @staticmethod
    def predict_proba_with(model, X_test):
        """Returns probabilities of each label."""

        probs = model.predict_proba(X_test)

        return probs


class AbstractSklearnModel(AbstractModel):
    @abstractmethod
    def create_model(self, emb_dim, num_classes):
        raise NotImplementedError('You have to create a model.')

    def fit(self, X_train, y_train, X_val, y_val):
        emb_dim = X_train.shape[1]  # embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes

        self.model = self.create_model(emb_dim, num_classes)

        param_grid = {'C': [0.001, 1, 1000]}
        clf = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=3,
            refit=True,
            # verbose=3
            verbose=0)

        clf.fit(X_train.numpy(), y_train.numpy())
        self.model = clf.best_estimator_
        return self.model

    def predict(self, X_test):
        """Returns predictions with class labels."""

        probs = self.model.predict_proba(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions



    def predict_proba(self, X_test):
        """Returns probabilities of each label."""

        probs = self.model.predict_proba(X_test)

        return probs


class AbstractNeuralNet(AbstractModel):
    def __init__(self, loss=losses.SparseCategoricalCrossentropy()):
        super().__init__()

        self.loss = loss
        self.oos_label = None  # used for CosFaceLofNN

    @abstractmethod
    def create_model(self, emb_dim, num_classes):
        raise NotImplementedError('You have to create a model.')

    def fit(self, X_train, y_train, X_val, y_val):
        emb_dim = X_train.shape[1]  # embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes

        self.model = self.create_model(emb_dim, num_classes)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                           loss=self.loss,
                           metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', mode="min", patience=10)
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


