#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn import preprocessing
import python_speech_features as spefeat
import librosa.feature as libfeat

from constants import AUDIO_DIR


def extract_mfcc(audio: np.array, samplerate: int):
    feature = spefeat.mfcc(audio, samplerate)

    return preprocessing.scale(feature)


def split_audio(
        audio: list, samplerate: int, th_duration: int = 10) \
        -> np.array:
    """
    Splits an audio in several extracts with a duration the nearest
    as possible to 'duration'

    :param audio: array representing the audio
    :param samplerate: number of values per seconds for audio
    :param th_duration: theoretical duration
    :return: array containing the splitted audio
    """

    total_duration = len(audio) / samplerate
    extract_number = int(total_duration / th_duration + .5)
    extract_len = len(audio) // extract_number
    maxi = extract_len * (extract_number - 1)

    return [
        audio[t:t + extract_len]
        for t in range(0, maxi, extract_len)
    ] + [audio[maxi:]]


def extract_features(
        data: pd.DataFrame, extract, train: bool = True,
        audio_dir: str = AUDIO_DIR) -> list:
    """

    :param data:
    :param extract:
    :param train:
    :param audio_dir:
    :return:
    """
    listed_features = []
    for index, row in data.iterrows():
        audio_name = row.loc["audio"]
        audio, samplerate = sf.read(audio_dir + audio_name)
        audio_extracts = split_audio(audio, samplerate)

        if train:
            audio_extracts = audio_extracts[0:1]

        for audio_extract in audio_extracts:
            # extract the features using the given extraction function
            listed_features.append(extract(audio_extract, samplerate))

    return listed_features


if __name__ == '__main__':
    audio, sp = sf.read("dev/audio/aahtm.flac")
    sp_audio = split_audio(audio, sp)
    for i in sp_audio:
        print(len(i))
