import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub


class UniversalSentenceEncoderFT:
    def __init__(self, model=None):
        if model is None:
            x = tf.keras.layers.Input(shape=[], dtype=tf.string)
            tmp_output = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=True)(x)
            self.model = tf.keras.models.Model(x, tmp_output)
        else:
            self.model = model
        self.model.summary()

    def metric_learning(self, X=None, y=None, saved_weights_path="./models_data/"):
        if X is not None and y is not None:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tfa.losses.TripletSemiHardLoss())
            self.model.fit(X, y, epochs=1, batch_size=128)
            self.model.save(saved_weights_path, include_optimizer=False)
        else:
            from tensorflow import keras
            self.model = keras.models.load_model(saved_weights_path)
        return self.model

    def __call__(self, *args, **kwargs):
        return self.model.predict(args)
