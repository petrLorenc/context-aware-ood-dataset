import numpy as np


# TP = predicted as OOD and true label is OOD
# TN = predicted as IN and true label is IN
# FP = predicted as OOD and true label is IN
# FN = predicted as IN and true label is OOD

# FAR = Number of accepted OOD sentences / Number of OOD sentences
# FAR = FN / (TP + FN)

# FRR = Number of rejected ID sentences / Number of ID sentences
# FRR = FP / (FP + TN)


class Testing:
    """Used to test the results of classification."""

    @staticmethod
    def test_threshold(model , X_test, y_test, oos_label, threshold: float, focus: str):
        if focus == "IND":
            accuracy, frr = 0, 0
            accuracy_out_of = 0
        elif focus == "OOD" or focus == "GARBAGE":
            recall, far = 0, 0
            recall_out_of = 0

        tp, tn, fp, fn = 0, 0, 0, 0

        pred_probs = model.predict_proba(
            X_test)  # returns numpy array

        pred_labels = np.argmax(pred_probs, axis=1)
        pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

        for pred_label, pred_similarity, true_label in zip(pred_labels, pred_similarities, y_test):
            if pred_similarity < threshold:
                pred_label = oos_label

            # the following set of conditions is the same for all testing methods
            if focus == "IND":
                if pred_label == true_label and true_label != oos_label:
                    accuracy += 1

                if pred_label != oos_label and true_label != oos_label:
                    tn += 1
                elif pred_label == oos_label and true_label != oos_label:
                    fp += 1
                accuracy_out_of += 1

            elif focus == "OOD" or focus == "GARBAGE":
                if pred_label == true_label and true_label == oos_label:
                    recall += 1
                    tp += 1
                elif pred_label != true_label and true_label == oos_label:
                    fn += 1
                recall_out_of += 1

        if focus == "IND":
            overall_accuracy = accuracy / accuracy_out_of * 100 # IND accuracy
            frr = fp / (fp + tn) * 100  # false recognition rate
            return {'accuracy': round(overall_accuracy, 1), 'frr': round(frr, 1)}

        elif focus == "OOD" or focus == "GARBAGE":
            overall_recall = recall / recall_out_of * 100 # ood recall
            far = fn / (tp + fn) * 100  # false acceptance rate
            return {'recall': round(overall_recall, 1), 'far': round(far, 1)}

    @staticmethod
    def test_illusionist(y_pred, y_test, oos_label, focus: str):
        if focus == "IND":
            accuracy, frr = 0, 0
            accuracy_out_of = 0
        elif focus == "OOD" or focus == "GARBAGE":
            recall, far = 0, 0
            recall_out_of = 0

        tp, tn, fp, fn = 0, 0, 0, 0

        pred_labels = y_pred

        for pred_label, true_label in zip(pred_labels, y_test):
            # the following set of conditions is the same for all testing methods
            if focus == "IND":
                if pred_label == true_label and true_label != oos_label:
                    accuracy += 1

                if pred_label != oos_label and true_label != oos_label:
                    tn += 1
                elif pred_label == oos_label and true_label != oos_label:
                    fp += 1
                accuracy_out_of += 1

            elif focus == "OOD" or focus == "GARBAGE":
                if pred_label == oos_label:
                    recall += 1
                    tp += 1
                elif pred_label != oos_label:
                    fn += 1
                recall_out_of += 1

        if focus == "IND":
            overall_accuracy = accuracy / accuracy_out_of * 100 # IND accuracy
            frr = fp / (fp + tn) * 100  # false recognition rate
            return {'accuracy': round(overall_accuracy, 1), 'frr': round(frr, 1)}

        elif focus == "OOD" or focus == "GARBAGE":
            overall_recall = recall / recall_out_of * 100 # ood recall
            far = fn / (tp + fn) * 100  # false acceptance rate
            return {'recall': round(overall_recall, 1), 'far': round(far, 1)}