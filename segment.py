#!/usr/bin/env python3
# coding: utf8

import pandas as pd

from constants import *


def create_dataframe(meta_name: str) -> pd.DataFrame:
    """

    :param meta_name:
    :return:
    """

    with open(meta_name) as meta_file:
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

    :param meta_data:
    :param nb_min:
    :return:
    """

    result = meta_data.loc[
        (meta_data[SPEAKER_NUMBER] == '1')
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

    :param meta_data:
    :return:
    """

    result = meta_data.loc[
        (meta_data[SPEAKER_NUMBER] == '1')
        ]

    # determines the number of sample for each speaker
    count = result[SPEAKER_ID].value_counts()

    # keeps only the samples if their speaker has enough other samples
    result = result.loc[
        result['speaker_id'].isin(count.index[count <= 5])
    ].reset_index(drop=True)

    return result


def split_dataset(dataset: pd.DataFrame, percent: int):
    """

    :param dataset:
    :param percent:
    :return:
    """

    dataset = dataset.reset_index(drop=True)
    i_min = dataset.index.min()
    i_max = dataset.index.max()
    mean = i_min + (i_max - i_min) * percent // 100 + 1

    return (
        dataset.iloc[i_min:mean],
        dataset.iloc[mean:]
    )
