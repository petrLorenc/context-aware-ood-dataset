import numpy as np

from abc import abstractmethod, ABCMeta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from models import AbstractModel


class AbstractSklearnModel(AbstractModel, metaclass=ABCMeta):
    def to_dict(self):
        return {

        }

    def fit(self, X_train, y_train, X_val, y_val):
        emb_dim = X_train.shape[1]  # embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes

        self.model = self.create_model(emb_dim, num_classes)

        # param_grid = {'C': [0.001, 1, 1000]}
        # clf = GridSearchCV(
        #     estimator=self.model,
        #     param_grid=param_grid,
        #     cv=3,
        #     refit=True,
        #     # verbose=3
        #     verbose=0)

        self.model.fit(X_train.numpy(), y_train.numpy())
        # self.model = clf.best_estimator_
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


class SklearnLogisticRegression(AbstractSklearnModel):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = LogisticRegression(max_iter=1500,
                                   penalty='l2',
                                   random_state=42,
                                   solver="lbfgs",
                                   class_weight="balanced",
                                   dual=False,
                                   multi_class="multinomial")
        return model


from sklearn.neural_network import MLPClassifier


class SklearnMLPClassifier(AbstractSklearnModel):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = MLPClassifier(hidden_layer_sizes=128,
                              activation='relu',
                              solver="adam",
                              learning_rate="adaptive",
                              early_stopping=True)
        return model


from sklearn.ensemble import RandomForestClassifier


class SklearnRandomForestClassifier(AbstractSklearnModel):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = RandomForestClassifier()
        return model
