import numpy as np  
import matplotlib.pyplot as plt 
import urllib.request
import tarfile
import pandas as pd

DATASET_FILE_NAME = "dataset.zip"


def get_dataset_contents(file_name: str) -> pd.DataFrame:
    """ Returns pandas dataframe of IMDB reviews and sentiment. """
    return pd.read_csv("dataset.zip")


def classify_text(text: str) -> list:
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
