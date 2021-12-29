import numpy as np
from sklearn.model_selection import GridSearchCV
from models.abstract_neural_net import AbstractModel
from sklearn.linear_model import LogisticRegression
from abc import abstractmethod


class AbstractOneClassModel(AbstractModel):
    def __init__(self, one_class_model):
        super().__init__()
        self.one_class_svm = one_class_model

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError('You have to create a model.')

    @abstractmethod
    def belong_to(self, one_class_svm, X_test) -> bool:
        raise NotImplementedError('You have to create a model.')


class OneClassSVMModel(AbstractOneClassModel):

    def create_model(self, emb_dim, num_classes):
        model = LogisticRegression(max_iter=850,
                                   penalty='l2',
                                   random_state=42,
                                   solver="lbfgs",
                                   class_weight="balanced",
                                   dual=False,
                                   multi_class="multinomial")
        return model

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

        self.one_class_svm.fit(X_train)

        return self.model, self.one_class_svm

    def belong_to(self, one_class_svm, X_test) -> bool:
        return one_class_svm.predict(X_test)[0] == 1

    def predict(self, X_test):
        return self.predict_with(self.model, X_test)

    def predict_proba(self, X_test):
        return self.predict_proba_with(self.model, X_test)

    def predict_with(self, model, X_test):
        """Returns predictions with class labels."""

        probs = model.predict_proba(X_test)
        predictions = np.argmax(probs, axis=1)

        return predictions

    def predict_proba_with(self, model, X_test):
        """Returns probabilities of each label."""

        probs = model.predict_proba(X_test)

        return probs
