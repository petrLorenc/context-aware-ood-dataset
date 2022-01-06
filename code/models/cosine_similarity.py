from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models import AbstractModel


class CosineSimilarity(AbstractModel):
    def create_model(self, emb_dim, num_classes):
        pass

    def to_dict(self):
        return {

        }

    def __init__(self):
        super().__init__()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    def predict(self, X_test):
        results = cosine_similarity(X_test, self.X_train)
        idxs = np.argmax(results, axis=1)
        predictions = self.y_train[idxs]

        return predictions

    def predict_proba(self, X_test):
        results = cosine_similarity(X_test, self.X_train)
        idxs = np.argmax(results, axis=1)
        probs = np.take_along_axis(results, indices=np.expand_dims(idxs, axis=1), axis=1).squeeze()
        predictions = self.y_train[idxs]  # class predictions (used as positions in sparse matrix)

        num_sents = len(X_test)  # number of sentences in test split
        num_classes = len(set(self.y_train))
        proba_mat = np.zeros((num_sents, num_classes))  # initialize sparse matrix

        for sent, prob, pos in zip(proba_mat, probs, predictions):
            sent[pos] = prob

        return proba_mat

    def threshold(self, val_predictions_labels, oos_label, value=0.5):
        """
            for pred, true_label in val_predictions_labels:
                pred_label = pred[0]
                similarity = pred[1]
        """
        return None
