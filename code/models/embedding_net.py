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
                 tfhub_handle_encoder,
                 tfhub_handle_preprocess,
                 seq_length,
                 use_masking_when_training,
                 epochs,
                 batch_size,
                 init_learning_rate):
        super().__init__()
        self.tfhub_handle_encoder = tfhub_handle_encoder
        self.tfhub_handle_preprocess = tfhub_handle_preprocess
        self.seq_length = seq_length
        self.preprocessor = None
        self.use_masking_when_training = use_masking_when_training

        self.epochs = epochs
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate

    def to_dict(self):
        return {
            "encoder":self.tfhub_handle_encoder,
            "preprocessor":self.preprocessor,
            "seq_length": self.seq_length,
            "use_masking_when_training": self.use_masking_when_training,
            "epoch": self.epochs,
            "batch_size": self.batch_size,
            "init_learning_rate": self.init_learning_rate
        }

    def create_model(self, num_classes):
        print(f'BERT model selected           : {self.tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')

        self.preprocessor = hub.load(self.tfhub_handle_preprocess)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')

        context_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="context")
        utterance_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="utterance")
        mask_input = tf.keras.layers.Input(shape=(num_classes,), dtype=tf.float32, name="mask_input")

        text_inputs = [context_input, utterance_input]

        tokenize = hub.KerasLayer(self.preprocessor.tokenize)
        tokenized_inputs = [tokenize(segment) for segment in text_inputs]

        seq_length = 128  # Your choice here.
        bert_pack_inputs = hub.KerasLayer(
            self.preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=seq_length))  # Optional argument.
        encoder_inputs = bert_pack_inputs(tokenized_inputs)

        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(num_classes, activation="softmax", name='classifier')(net)

        # model_output = tf.keras.layers.Multiply()([net, mask_input])

        # not a proper Softmax (TODO: try another normalization techniques)

        action_probs_masked = tf.keras.layers.Multiply()([net, mask_input])
        layer = tf.keras.layers.Lambda(lambda x: x / tf.keras.backend.sum(x, axis=1)[:, None])
        model_output = layer(action_probs_masked)

        self.model = tf.keras.Model(inputs={"context_input": context_input, "utterance_input": utterance_input, "mask_input": mask_input}, outputs=model_output)
        self.model.summary()

    def fit(self, X_train, y_train, X_val, y_val):
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = tf.keras.metrics.CategoricalAccuracy()

        steps_per_epoch = len(y_train) // self.batch_size
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.05 * num_train_steps)

        optimizer = optimization.create_optimizer(init_lr=self.init_learning_rate,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

        print("Starting preprocessing data")
        # _X_train = self.preprocessor.bert_pack_inputs([self.preprocessor.tokenize(X_train[0]), self.preprocessor.tokenize(X_train[1])], seq_length=self.seq_length)
        # _X_val = self.preprocessor.bert_pack_inputs([self.preprocessor.tokenize(X_val[0]), self.preprocessor.tokenize(X_val[1])], seq_length=self.seq_length)
        _X_train = {"context_input": tf.convert_to_tensor(X_train[0]), "utterance_input": tf.convert_to_tensor(X_train[1])}
        _X_val = {"context_input": tf.convert_to_tensor(X_val[0]), "utterance_input": tf.convert_to_tensor(X_val[1])}
        print("Data preprocessed")

        if self.use_masking_when_training:
            _X_train["mask_input"] = X_train[2]
            _X_val["mask_input"] = X_val[2]
        else:
            _X_train["mask_input"] = np.ones_like(X_train[2])
            _X_val["mask_input"] = np.ones_like(X_val[2])

        try:
            history = self.model.fit(x=_X_train, y=y_train,
                                     validation_data=(_X_val, y_val),
                                     epochs=self.epochs, batch_size=self.batch_size)
        except KeyboardInterrupt:
            history = None
            print("Training stopped by CTRL+C")

        return history

    def predict_proba(self, X_test, mask=None):
        # _X_test = self.preprocessor.bert_pack_inputs([self.preprocessor.tokenize(X_test[0]), self.preprocessor.tokenize(X_test[1])],
        #                                              seq_length=self.seq_length)
        _X_test = {"context_input": tf.convert_to_tensor(X_test[0]), "utterance_input": tf.convert_to_tensor(X_test[1]), "mask_input": X_test[2]}
        return self.model.predict(_X_test)
