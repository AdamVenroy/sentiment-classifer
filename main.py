import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd
import re
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
    shutil.rmtree(remove_dir)

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, 
        subset='training', 
        seed=SEED
        )
    
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, 
        subset='validation', 
        seed=SEED
    )

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test', 
        batch_size=BATCH_SIZE
    )

    return raw_train_ds, raw_val_ds, raw_test_ds
    






def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Turns all text lowerspace, removes html tags"""
    # Standardization
    df["review"] = df["review"].apply(lambda x: x.lower())
    df["review"] = df["review"].apply(lambda x: re.sub(r'<[^>]*>', ' ', x))
    df["review"] = df["review"].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['target'] = df['sentiment'].str.contains('positive')
    df['target'] = df['target'].apply(int)
    return df


def vectorize_data(df: pd.DataFrame) -> tf.Tensor:
    """ Given a standardized dataframe, returns a vectorized dataframe """
    vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=MAX_FEATURES, 
                                                        output_mode='int', output_sequence_length=SEQUENCE_LENGTH)
    vectorize_layer.adapt(df['review'])
    return vectorize_layer(df['review'])


def create_model(df: pd.DataFrame) -> tf.keras.Model:
    """ Creates and trains the model based on vectorized data. """
    training_input_data = vectorize_data(df)
    target = df.pop('target')
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(training_input_data)

    training_output_data = target
    print(training_input_data)
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(125, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
    )

    
    model.fit(training_input_data, training_output_data, epochs=50)

    return model


def test_model(df: pd.DataFrame, model: tf.keras.Model) -> None:
    """ Prints data from evaluating test."""
    x = vectorize_data(df)
    y = df.pop('target')
    results = model.evaluate(x, y)
    print(f"loss: {results[0]}, accuracy: {results[1]}")


def classify_text() -> list:
    """ Given a string returns a list where the first value is 
    the probality that the text has a positive sentiment, 
    and the second value is the probability that the text has 
    a negative sentiment. """
    pass


def main():
    """ Main function """
    pass

if __name__ == "__main__":
    a, b, c = get_data_from_dataset_folder()