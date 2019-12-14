#!/usr/bin/env python3
# coding: utf8

import numpy as np
import scipy as scp
import pandas as pd
from sklearn import svm
from sklearn import mixture
from sklearn import preprocessing
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import python_speech_features as spefeat
import librosa.feature as libfeat

from typing import List


# columns
AUDIO = 'audio'
SPEAKER_ID = 'speaker_id'
GENDER = 'gender'
MICRO = 'micro'
SESSION_ID = 'session_id'
START = 'start'
END = 'end'
SPEAKER_NUMBER = 'speaker_number'
LABELS = 'labels'
DEGRADATION = 'degradation'

AUDIO_DIRECTORY = 'dev/'

PROPORTION_COLUMNS = [GENDER, MICRO, DEGRADATION, SPEAKER_NUMBER]

COLORS = ["#4384DB", "#E14744", "#BDE267", "#895FBD", "#52C1E8"]


def display_audio_extract(audio_name: str = 'aaoao.flac'):
    data, samplerate = sf.read(AUDIO_DIRECTORY + audio_name)

    alpha = .5
    window_size = .025
    frame = int(samplerate * window_size)
    start = int(.12 * samplerate)
    new_start = int(.11 * samplerate)

    data = data[start:start + samplerate * 5]
    time = np.array([x / samplerate for x in range(len(data))])
    ham = (1 - alpha) + alpha * \
        np.cos(np.pi * (2 * time / window_size + 1))

    corrected_data = ham[:frame] * data[:frame]

    data_fft = np.fft.fft(data)
    data_freqs = np.fft.fftfreq(len(data)) * samplerate

    corrected_data_fft = np.fft.fft(corrected_data)
    corrected_data_freqs = \
        np.fft.fftfreq(len(corrected_data)) * samplerate

    plt.figure(0)
    plt.subplot(211)
    plt.plot(time, data)
    plt.xlim(time[0], time[-1])
    plt.ylabel("amplitude")

    plt.subplot(212)
    plt.specgram(data, Fs=samplerate)
    plt.xlabel("time (s)")
    plt.ylabel("frequency")

    plt.figure(1)
    plt.subplot(211)
    plt.plot(time[:frame], data[:frame])
    plt.ylabel("amplitude")

    plt.subplot(212)
    plt.plot(time[:frame], corrected_data)
    plt.ylabel("amplitude")
    plt.xlabel("time (s)")

    plt.figure(2)
    plt.subplot(211)
    plt.plot(
        data_freqs[:len(data_freqs) // 2],
        abs(data_fft[:len(data_freqs) // 2])
    )
    plt.ylabel("amplitude")

    plt.subplot(212)
    plt.plot(
        corrected_data_freqs[:len(corrected_data_freqs) // 2],
        abs(corrected_data_fft[:len(corrected_data_freqs) // 2])
    )
    plt.ylabel("amplitude")
    plt.xlabel("frequency")

    plt.figure(3)
    plt.subplot(211)
    plt.plot(
        time[new_start: new_start + 2 * frame],
        data[new_start: new_start + 2 * frame]
    )
    plt.ylabel("amplitude")
    plt.xlabel("time (s)")

    plt.show()

    return data, samplerate


def append_proportion(
        ax1: plt.Subplot, ax2: plt.Subplot, series: pd.Series, name: str
) -> list:
    values = series.values
    labels = series.keys()

    wedges, _, autotexts = ax2.pie(
        values, autopct='%1.1f%%', startangle=90,
        colors=COLORS, counterclock=False
    )

    ax1.legend(
        wedges, labels,
        title=name,
        loc="center"
    )

    ax2.axis('equal')
    ax1.axis('off')

    return autotexts


def display_proportions(data: pd.DataFrame, properties: List[str]):
    fig = plt.figure(figsize=(10, 8))
    container = gridspec.GridSpec(2, 2)

    for i in range(len(properties)):
        content = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=container[i], height_ratios=[1, 3]
        )

        ax1 = plt.Subplot(fig, content[0])
        ax2 = plt.Subplot(fig, content[1])

        autotexts = append_proportion(
            ax1, ax2, pd.Series(
                data[properties[i]].value_counts()
            ), properties[i]
        )
        plt.setp(autotexts, size=8, weight="bold")

        fig.add_subplot(ax1)
        fig.add_subplot(ax2)

    fig.show()


def create_dataframe(meta_name: str = 'dev/keys/meta.lst') \
        -> pd.DataFrame:
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


def extract_data(meta_data: pd.DataFrame) -> pd.DataFrame:
    result = meta_data.loc[
        (meta_data[SPEAKER_NUMBER] == '1') &
        (meta_data[DEGRADATION] <= 2)
        ]

    # determines the number of sample for each speaker
    count = result[SPEAKER_ID].value_counts()

    # keeps only the samples if their speaker has enough other samples
    result = result.loc[
        result['speaker_id'].isin(count.index[count >= 5])
    ].reset_index(drop=True)

    return result


if __name__ == "__main__":
    # data, samplerate = display_audio_extract()
    # mfcc_lib = libfeat.mfcc(y=data, sr=samplerate)
    # mfcc_spe = spefeat.mfcc(data, samplerate)

    dev_data = extract_data(create_dataframe('dev/keys/meta.lst'))
    # eval_data = create_dataframe('eval/keys/meta.lst')

    dev_data_speaker = dev_data[SPEAKER_ID].values

    characteristics = []
    background = []
    classes = []
    features = np.asarray(())
    tests = np.asarray(())

    speaker = dev_data.iloc[0][SPEAKER_ID]

    for index, row in dev_data.iterrows():
        audio_name, speaker_id = row.loc[[AUDIO, SPEAKER_ID]]
        data, samplerate = sf.read(AUDIO_DIRECTORY + audio_name)
        # if speaker_id == speaker:
        #     characteristics.append(libfeat.mfcc(y=data, sr=samplerate))
        # else:
        #     background.append(libfeat.mfcc(y=data, sr=samplerate))
        vector = spefeat.mfcc(data, samplerate)
        vector = preprocessing.scale(vector)
        if speaker_id == speaker:
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        else:
            if tests.size == 0:
                tests = vector
            else:
                tests = np.vstack((tests, vector))
        classes.append(1 if speaker_id == speaker else 0)

    print("gmm starts")

    gmm1 = mixture.GaussianMixture(
        n_components=3, covariance_type='diag', n_init=3
    )

    gmm2 = mixture.GaussianMixture(
        n_components=8, covariance_type='diag', n_init=3
    )

    gmm1.fit(features[:3])
    gmm2.fit(tests[:8])

    print(gmm1.score(features[3:]))
    print(gmm2.score(features[3:]))

    print(gmm1.score(tests[8:]))
    print(gmm2.score(tests[8:]))

    # clf = svm.SVC()
    # clf.fit(characteristics[:10], classes[:10])
    # clf.predict(characteristics[10:])

    # with open('dev/lists/enroll-core.lst') as core_file:
    #     core = pd.DataFrame([
    #         line.replace('\n', '').split(' ')[1] for line in core_file],
    #         columns=['audio']
    #     )

    # result = core.merge(meta_data, on='audio', how='left')

    extracted_result = extract_data(create_dataframe())

    # display_proportions(extracted_result, PROPORTION_COLUMNS)

    # for line in extracted_result:
    #     print(line[AUDIO])
