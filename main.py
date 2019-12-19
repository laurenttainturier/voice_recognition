#!/usr/bin/env python3
# coding: utf8

import matplotlib.pyplot as plt

from segment import *
from extract import *
from classify import *


dev_dataset = extract_dataset(create_dataframe('dev/keys/meta.lst'))
eval_dataset = extract_dataset(create_dataframe('eval/keys/meta.lst'))


if __name__ == "__main__":

    original_dataset = create_dataframe('dev/keys/meta.lst')

    dataset = extract_dataset(original_dataset)
    ubm_dataset = extract_background_dataset(original_dataset)

    ubm_train_dataset, ubm_test_dataset = split_dataset(ubm_dataset, 30)
    ubm_train_features = extract_features(ubm_train_dataset, extract_mfcc)
    ubm_test_features = extract_features(ubm_test_dataset, extract_mfcc)

    gmm_ubm = train_gmm(ubm_train_features)

    for i, speaker in enumerate(sorted(dataset[SPEAKER_ID].unique())):
        spk_dataset = dataset.loc[dataset[SPEAKER_ID] == speaker]
        spk_train, spk_test = split_dataset(spk_dataset, 30)
        spk_train_features = extract_features(spk_train, extract_mfcc)
        spk_test_features = extract_features(
            spk_test, extract_mfcc, train=False
        )

        gmm_spk = train_gmm(spk_train_features)

        spk = []
        ubm = []

        for test in spk_test_features:
            spk.append(gmm_spk.score(test))
            ubm.append(gmm_ubm.score(test))

        diff = np.array(spk) - np.array(ubm)

        fig = plt.figure(i)
        plt.hist(diff, alpha=0.5, label='genuine')
        plt.legend()

        oth_dataset = dataset.loc[dataset[SPEAKER_ID] != speaker]
        oth_test = extract_features(
            oth_dataset, extract_mfcc, train=False
        )

        spk = []
        ubm = []

        for test in oth_test:
            spk.append(gmm_spk.score(test))
            ubm.append(gmm_ubm.score(test))

        diff = np.array(spk) - np.array(ubm)

        plt.hist(diff, alpha=0.5, label='impostor')
        plt.legend()
        plt.savefig(f'picture/{i}_similarity.png')
        plt.close(fig)
