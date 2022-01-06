import tensorflow as tf


class TensorflowBERT:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    def __call__(self, *args, **kwargs):
        _X = {"context_input": tf.convert_to_tensor(args[0][0]), "utterance_input": tf.convert_to_tensor(args[0][1])}
        return self.model.predict(_X)
