import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot

import random
from metric_learning.contrastive_learning import UniversalSentenceEncoderFT

size = 1000

if __name__ == '__main__':
    # embedding_model = UniversalSentenceEncoderFT()
    # embedding_model.metric_learning(X=[f"sentence {x}" for x in range(size)], y=[random.randint(0, 5) for _ in range(size)],
    #                                 saved_weights_path="/media/petrlorenc/Data/universal-sentence-encoder_fine/")
    # print(embedding_model(["Sentence sfsd"]))

    x = tf.keras.layers.Input(shape=[], dtype=tf.string)
    tmp_output = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=True)(x)
    model = tf.keras.models.Model(x, tmp_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())
    model.fit([f"sentence {x}" for x in range(size)], y=[random.randint(0, 5) for _ in range(size)], epochs=1, batch_size=128)

    model.save("/media/petrlorenc/Data/universal-sentence-encoder_fine/", include_optimizer=False)
    #
    # # Helper function uses `quantize_annotate_layer` to annotate that only the
    # # Dense layers should be quantized.
    # def apply_quantization_to_dense(layer):
    #     if isinstance(layer, tf.keras.layers.Dense):
    #         return tfmot.quantization.keras.quantize_annotate_layer(layer)
    #     return layer
    #
    #
    # # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense`
    # # to the layers of the model.
    # annotated_model = tf.keras.models.clone_model(
    #     model,
    #     clone_function=apply_quantization_to_dense,
    # )
    # quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    #
    # quant_aware_model.save("/media/petrlorenc/Data/universal-sentence-encoder_fine/", include_optimizer=False)
