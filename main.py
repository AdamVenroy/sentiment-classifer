import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd
import re
import string
import os
import shutil
import tensorflow as tf

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SEQUENCE_LENGTH = 250 # Number of maximum tokens
MAX_FEATURES = 10000 # Number of perceptrons on input layer
AMOUNT_OF_TRAINING_DATA = 25000
BATCH_SIZE = 32
SEED = 42

def download_dataset_file():
    """ Downloads the file from the URL and unzips it """
    tf.keras.utils.get_file("aclImdb_v1", URL,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')


def get_data_from_dataset_folder():
    """ Looks into the directorys from the unzipped file and returns the training,
    validation and testing datasets."""
    
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    try:
        shutil.rmtree(remove_dir)
    except FileNotFoundError:
        print("unsupervised data already removed.")

    raw_training_dataset = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, 
        subset='training', 
        seed=SEED
        )
    
    raw_validation_dataset = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, 
        subset='validation', 
        seed=SEED
    )

    raw_testing_dataset = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test', 
        batch_size=BATCH_SIZE
    )

    return raw_training_dataset, raw_validation_dataset, raw_testing_dataset


def custom_standardization(input_data):
  """ Standardisation function for stripping reviews into tokens for 
  vectorization"""
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


def create_vectorization_layer(raw_training_dataset):
    """ Creates vectorization layer for NN and adapts it to training text."""
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_FEATURES,
        output_sequence_length=SEQUENCE_LENGTH
    )

    train_text = raw_training_dataset.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    return vectorize_layer


def vectorize_text(text, label, vectorize_layer):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

def create_model(raw_training_dataset, raw_validation_dataset) -> tf.keras.Model:
    """ Creates and trains the model. """
    vectorization_layer = create_vectorization_layer(raw_training_dataset)
    training_dataset = raw_training_dataset.map(lambda x, y: vectorize_text(x, y, vectorization_layer))
    validation_dataset = raw_validation_dataset.map(lambda x, y: vectorize_text(x, y, vectorization_layer))

    AUTOTUNE = tf.data.AUTOTUNE
    training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_FEATURES, 16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)]
    )

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=10)
    
    return model


def classify_text() -> list:
    """ Given a string returns a list where the first value is 
    the probality that the text has a positive sentiment, 
    and the second value is the probability that the text has 
    a negative sentiment. """
    pass


def main():
    """ Main function """
    pass
def vectorize_text(text, label, vectorize_layer):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label
if __name__ == "__main__":
    raw_train_ds, raw_val_ds, raw_test_ds = get_data_from_dataset_folder()
    m = create_model(raw_train_ds, raw_val_ds)
