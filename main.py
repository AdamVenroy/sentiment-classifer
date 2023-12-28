import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd
import re
from collections import Counter
import tensorflow as tf

DATASET_FILE_NAME = "dataset.zip"
SEQUENCE_LENGTH = 250 # Number of maximum tokens
MAX_FEATURES = 10000 # Number of perceptrons on input layer
AMOUNT_OF_TRAINING_DATA = 25000

def get_dataset_contents(file_name: str) -> pd.DataFrame:
    """ Returns pandas dataframe of IMDB reviews and sentiment. """
    return pd.read_csv(file_name)


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
    print("Getting dataset...")
    x = get_dataset_contents(DATASET_FILE_NAME)
    print("Standardizing data...")
    x = standardize_data(x)
    training_data = x[:AMOUNT_OF_TRAINING_DATA]
    testing_data = x[AMOUNT_OF_TRAINING_DATA:]
    model = create_model(x)
    test_model(testing_data, model)

