#!/usr/bin/env python3
# coding: utf8

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn import preprocessing
import python_speech_features as spefeat
import librosa.core as libcore

from tools import *
from constants import AUDIO_DIR, AUDIO, SPEAKER_ID
from segment import segment_audio, split_in_windows


def extract_with_mfcc(audio: np.array, samplerate: int):
    """
    Extracts audio characteristics using LPC

    :param audio: (np.array) audio to be extracted
    :param samplerate: (int) corresponding to the samplerate of audio
    :return: characteristics of the signal
    """

    features = spefeat.mfcc(audio, samplerate)

    return preprocessing.scale(features)


def extract_with_lpc(audio: np.array, samplerate: int):
    """
    Extracts audio characteristics using LPC

    :param audio: (np.array) audio to be extracted
    :param samplerate: (int) corresponding to the samplerate of audio
    :return: characteristics of the signal
    """

    windows = split_in_windows(audio, samplerate)

    # for each window, get the coefficients of the LPC
    features = np.array([
        libcore.lpc(window, 12) for window in windows
    ])

    return preprocessing.scale(features)


def extract_with_plp(audio: np.array, _: int):
    """
    Extracts audio characteristics using PLP

    :param audio: (np.array) audio to be extracted
    :param _: (int) corresponding to the samplerate of audio
    :return: characteristics of the signal
    """

    pass


@get_function_duration
@get_function_memory_consumption
def extract_features(
        data: pd.DataFrame, extract, multi: bool = True,
        audio_dir: str = AUDIO_DIR) -> (list, list):
    """
    Generic function to perform feature extraction,
    independently of the extraction method

    :param data: (pd.DataFrame) contains the detail of each sample
    :param extract: (Function) extraction method to use
    :param multi: (bool) specify if several samples can be extract
        from one audio file
    :param audio_dir: (str) directory where the audio files are located
    :return: (features:list, labels:list)
    """
    features = []
    speakers = []

    for index, row in data.iterrows():
        audio_name = row.loc[AUDIO]
        speaker = row.loc[SPEAKER_ID]
        audio, samplerate = sf.read(audio_dir + audio_name)
        audio_extracts = segment_audio(audio, samplerate)

        if not multi:
            audio_extracts = audio_extracts[0:1]

        for audio_extract in audio_extracts:
            # extract the features using the given extraction function
            features.append(extract(audio_extract, samplerate))
            speakers.append(speaker)

    return features, speakers


if __name__ == '__main__':
    audio, sp = sf.read("database/dev/audio/aahtm.flac")
    sp_audio = segment_audio(audio, sp)
    lpc = extract_with_lpc(audio, sp)
