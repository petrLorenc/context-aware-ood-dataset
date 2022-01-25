import os
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."  # Project root

# local
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # remote
    TENSORFLOW_ROOT = "/home/lorenpe2/USE-embeddings/"
    BERT_ROOT = "/home/lorenpe2/BERT_embeddings/"
    HUGGING_FACE_ROOT = "sentence-transformers"  # will download from hugging face
else:
    TENSORFLOW_ROOT = "/media/petrlorenc/Data/"
    BERT_ROOT = "/media/petrlorenc/Data/"
    HUGGING_FACE_ROOT = "/home/lorenpe2/BERT_embeddings/hugging_face_models"
