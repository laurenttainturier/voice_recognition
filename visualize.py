#!/usr/bin/env python3
# coding: utf8
import numpy as np
import pandas as pd
import soundfile as sf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from constants import AUDIO_DIR, COLORS


def display_audio_extract(audio_name: str = 'aaoao.flac'):
    data, samplerate = sf.read(AUDIO_DIR + audio_name)

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


def add_proportion(
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


def display_proportions(data: pd.DataFrame, properties: list):
    fig = plt.figure(figsize=(10, 8))
    container = gridspec.GridSpec(2, 2)

    for i, propertie in enumerate(properties):
        content = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=container[i], height_ratios=[1, 3]
        )

        ax1 = plt.Subplot(fig, content[0])
        ax2 = plt.Subplot(fig, content[1])

        autotexts = add_proportion(
            ax1, ax2, pd.Series(
                data[propertie].value_counts()
            ), propertie
        )
        plt.setp(autotexts, size=8, weight="bold")

        fig.add_subplot(ax1)
        fig.add_subplot(ax2)

    fig.show()

