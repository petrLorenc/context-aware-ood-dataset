from abc import ABC, abstractmethod, ABCMeta
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


class AbstractEmbeddingModel(ABC):
    def __init__(self):
        np.random.seed(7)  # set seed in order to have reproducible results

        self.model = None
        self.model_name = type(self).__name__

    @abstractmethod
    def create_model(self, num_classes):
        raise NotImplementedError('You have to create a model.')

    @abstractmethod
    def predict_proba(self, X_test, mask=None):
        raise NotImplementedError('This function needs to be implemented.')

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError('This function needs to be implemented.')


import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer


class BERTEmbeddingModel(AbstractEmbeddingModel):

    def __init__(self,
                 tfhub_handle_encoder="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
                 tfhub_handle_preprocess="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"):
        super().__init__()
        self.tfhub_handle_encoder = tfhub_handle_encoder
        self.tfhub_handle_preprocess = tfhub_handle_preprocess

    def create_model(self, num_classes):
        print(f'BERT model selected           : {self.tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(num_classes, activation="softmax", name='classifier')(net)
        self.model = tf.keras.Model(text_input, net)
        self.model.summary()

    def fit(self, X_train, y_train, X_val, y_val):
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = tf.metrics.Accuracy()

        epochs = 5
        batch_size = 64
        steps_per_epoch = len(y_train) // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.05 * num_train_steps)

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

        history = self.model.fit(x=np.array(X_train).reshape(-1, 1), y=y_train,
                                 validation_data=(np.array(X_val).reshape(-1, 1), y_val),
                                 epochs=epochs, batch_size=batch_size)
        return history

    def predict_proba(self, X_test, mask=None):
        p = self.model.predict(X_test)[0] * mask
        return np.exp(p) / sum(np.exp(p))
