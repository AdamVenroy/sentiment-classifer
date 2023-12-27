import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd
import re
from collections import Counter

DATASET_FILE_NAME = "dataset.zip"
MAXIMUM_SEQ_LENGTH = 10000 # Number of maximum tokens
MAXIMUM_INPUT_POINTS = 250 # Number of perceptrons on input layer


def get_dataset_contents(file_name) -> pd.DataFrame:
    """ Returns pandas dataframe of IMDB reviews and sentiment. """
    return pd.read_csv(file_name)


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Turns all text lowerspace, removes html tags"""
    # Standardization
    df["review"] = df["review"].apply(lambda x: x.lower())
    df["review"] = df["review"].apply(lambda x: re.sub(r'<[^>]*>', ' ', x))
    df["review"] = df["review"].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return df


def vectorize_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Given a standardized dataframe, returns a vectorized dataframe """
    df = df[df["review"].map(lambda x: len(x.split())) <= MAXIMUM_SEQ_LENGTH]
    input_tokens = [tup[0] for tup in Counter(" ".join(df["review"]).split()).most_common(MAXIMUM_INPUT_POINTS)]
    series_list = []

    for token in input_tokens:
        series_list.append(df["review"].apply(lambda x: int(token in x)))

    vectorized_df = pd.concat(series_list, axis=1, keys=input_tokens)
    vectorized_df['target'] = df['sentiment'].str.contains('positive')
    vectorized_df['target'] = vectorized_df['target'].apply(int)
    return vectorized_df


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
    x = get_dataset_contents(DATASET_FILE_NAME)
    x = standardize_data(x)
    x = vectorize_data(x)
    print(x)
