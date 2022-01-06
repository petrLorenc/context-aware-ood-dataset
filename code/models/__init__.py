from abc import ABC, abstractmethod

import numpy as np


class AbstractModel(ABC):
    def __init__(self):
        np.random.seed(7)  # set seed in order to have reproducible results
        self.model = None
        self.model_name = type(self).__name__

    @abstractmethod
    def create_model(self, emb_dim, num_classes):
        raise NotImplementedError('You have to create a model.')

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError('This function needs to be implemented.')

    @abstractmethod
    def predict_proba(self, X_test):
        raise NotImplementedError('This function needs to be implemented.')

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError('This function needs to be implemented.')

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError('This function needs to be implemented.')

    def predict_with(self, model, X_test):
        """Returns predictions with class labels."""

        probs = model.predict_proba(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions

    def predict_proba_with(self, model, X_test):
        """Returns probabilities of each label."""

        probs = model.predict_proba(X_test)

        return probs