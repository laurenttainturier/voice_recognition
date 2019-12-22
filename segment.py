#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pandas as pd

from constants import *


def segment_audio(
        audio: list, samplerate: int, th_duration: int = 10) \
        -> np.array:
    """
    Splits an audio in several extracts with a duration
    the nearest as possible to 'duration'

    :param audio: array representing the audio
    :param samplerate: number of values per seconds for audio
    :param th_duration: theoretical duration (in s)
    :return: array containing the splitted audio
    """

    total_duration = len(audio) / samplerate
    extract_number = int(total_duration / th_duration + .5)
    extract_len = len(audio) // extract_number
    maxi = extract_len * (extract_number - 1)

    return [
        audio[t:t + extract_len]
        for t in range(0, maxi, extract_len)
    ]


def split_in_windows(
        audio: np.array, samplerate: int,
        size: int = 25, overlap: int = 10) -> np.array:
    """
    Splits the audio in a list of window

    :param audio: (np.array)
    :param samplerate: (int)
    :param size: (int) size of each window (in ms)
    :param overlap: (int) overlap between each window (in ms)
    :return: (list) containing the windows
    """

    window_size = samplerate * size // 1000
    window_overlap = samplerate * overlap // 1000
    frame_max = (len(audio) // window_overlap - 1) * window_overlap

    return [
        audio[x:x + window_size]
        for x in range(0, frame_max, window_overlap)
    ]


def create_dataframe(metadata_location: str) -> pd.DataFrame:
    """
    Creates a dataframe from the file containing
    the metadata for the database

    :param metadata_location: (str) location of the metadata file
    :return: (pd.DataFrame)
    """

    with open(metadata_location) as meta_file:
        meta_data = pd.DataFrame([
            # keeps only the 10 first elements of each row
            line.replace(', ', ',').replace('\n', '').split(' ')[:10]
            for line in meta_file
        ], columns=[
            AUDIO, SPEAKER_ID, GENDER, MICRO, SESSION_ID,
            START, END, SPEAKER_NUMBER, LABELS, DEGRADATION
        ])

    # extract only the data matching certain conditions
    meta_data[DEGRADATION] = meta_data[DEGRADATION].astype(int)

    return meta_data


def extract_dataset(meta_data: pd.DataFrame, nb_min=9) -> pd.DataFrame:
    """
    Extracts dataset from 'meta_data' with the samples
    for a given speaker if they are more than a given number

    :param meta_data: (pd.DataFrame)
    :param nb_min: (int) number of sample min for each speaker
    :return: (pd.DataFrame) dataset
    """

    result = meta_data.loc[
        (meta_data[SPEAKER_NUMBER] == '1') &
        (meta_data[DEGRADATION] <= 4)
    ]

    # determines the number of sample for each speaker
    count = result[SPEAKER_ID].value_counts()

    # keeps only the samples if their speaker has enough other samples
    result = result.loc[
        result['speaker_id'].isin(count.index[count >= nb_min])
    ].reset_index(drop=True)

    return result


def extract_background_dataset(meta_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a background dataset from 'meta_data' with
    undernourished samples for a given speaker

    :param meta_data: (pd.DataFrame)
    :return: (pd.DataFrame) background dataset
    """

    result = meta_data.loc[
        (meta_data[SPEAKER_NUMBER] == '1') &
        (meta_data[DEGRADATION] <= 4)
    ]

    # determines the number of sample for each speaker
    count = result[SPEAKER_ID].value_counts()

    # keeps only the samples if their speaker has enough other samples
    result = result.loc[
        result['speaker_id'].isin(count.index[count <= 5])
    ].reset_index(drop=True)

    return result


def split_dataset(dataset: pd.DataFrame, percent: int) \
        -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataset in two, the first will contains
    around 'percent' percent of the original dataset.
    Ensure that each speaker is represented with the same proportion

    :param dataset: (pd.DataFrame) to be splitted
    :param percent: (int) between 0 and 100
    :return: (pd.DataFrame, pd.DataFrame) the new datasets
    """

    dataset = dataset.reset_index(drop=True)

    speakers = dataset[SPEAKER_ID].unique()
    train_indexes = []
    test_indexes = []

    for speaker in speakers:
        indexes = dataset.index[dataset[SPEAKER_ID] == speaker].tolist()
        mean = len(indexes) * percent // 100 + 1
        train_indexes += indexes[:mean]
        test_indexes += indexes[mean:]

    return (
        dataset.iloc[train_indexes, :].reset_index(drop=True),
        dataset.iloc[test_indexes, :].reset_index(drop=True)
    )
