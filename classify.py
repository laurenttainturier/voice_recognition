#!/usr/bin/env python3
# coding: utf8

import matplotlib.pyplot as plt

from sklearn import mixture, svm

from tools import *


class Classifier:
    def train(self, features: list, labels: list):
        """
        Trains the classifier using the given features

        :param features: (list) list of the features for each sample
        :param labels: (list) corresponding label for each sample
        :return:
        """

        print("not implemented")

    def train_background(self, features: list):
        """
        Trains the classifier using the background data

        :param features: (list) list of the features for each sample
        :return:
        """
        print("not implemented")

    def classify(self, features: list, labels: list):
        """
        Generates results for the classifier.
        'features' and 'labels' must have the same dimension

        :param features: (list) list of the features for each sample
        :param labels: (list) corresponding label for each sample
        """

        print("not implemented")


class GMMClassifier(Classifier):
    def __init__(self):
        self.gmms = {}

    @get_function_duration
    @get_function_memory_consumption
    def train(self, features: list, labels: list):
        """
        Trains gmm using the given features

        :param features: (list) list of the features for each sample
        :param labels: (list) corresponding label for each sample
        :return:
        """
        np_labels = np.array(labels)
        np_features = np.array(features)

        for i, label in enumerate(np.unique(np_labels)):
            self.gmms[label] = mixture.GaussianMixture(
                n_components=3, covariance_type='diag', n_init=3
            )

            indexes, *_ = np.where(np_labels == label)

            self.gmms[label].fit(
                np.vstack(np.take(np_features, indexes))
            )

    @get_function_duration
    @get_function_memory_consumption
    def train_background(self, features: list):
        """
        Trains GMM using the background data

        :param features: (list) list of the features for each sample
        :return:
        """
        self.gmms['ubm'] = mixture.GaussianMixture(
            n_components=3, covariance_type='diag', n_init=3
        )

        self.gmms['ubm'].fit(np.vstack(features))

    @get_function_duration
    @get_function_memory_consumption
    def classify(self, features: list, labels: list):
        """
        Generates results for the classifier GMM
        'features' and 'labels' must have the same dimension

        :param features: (list) list of the features for each sample
        :param labels: (list) corresponding label for each sample
        :return:
        """
        if not self.gmms:
            print("The classifier must first be trained "
                  "before being able to classify")

        fig = plt.figure()
        speakers = np.unique(np.array(labels))

        for j, speaker_to_identify in enumerate(speakers):

            genuine_scores = []
            impostor_scores = []

            for i, feature in enumerate(features):
                current_speaker = labels[i]
                score = \
                    self.gmms[speaker_to_identify].score(feature) - \
                    self.gmms['ubm'].score(feature)

                if current_speaker == speaker_to_identify:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

            threshold_min = min(genuine_scores + impostor_scores)
            threshold_max = max(genuine_scores + impostor_scores)

            fig = plt.figure()
            plt.hist(genuine_scores, alpha=.5, label='genuine')
            plt.hist(impostor_scores, alpha=.5, label='impostor')
            plt.legend()
            plt.xlabel("Dissimilarity")
            plt.ylabel("Number of samples")
            plt.title(
                "Distribution of genuine and imposter classification"
            )
            plt.savefig(
                f'picture/score/lpc_gmm/{speaker_to_identify}_res.png'
            )
            plt.close(fig)

            far = []
            frr = []

            for threshold in np.linspace(
                    threshold_min, threshold_max, 200):

                tp = 0
                tn = 0

                for genuine_score in genuine_scores:
                    if genuine_score >= threshold:
                        tp += 1

                for impostor_score in impostor_scores:
                    if impostor_score < threshold:
                        tn += 1

                far.append(1 - (tn / len(impostor_scores)))
                frr.append(1 - (tp / len(genuine_scores)))

            fig = plt.figure()
            plt.plot(far, frr)
            plt.xlabel("False Accept Rate(FAR), in %")
            plt.ylabel("False Rejection Rate(FRR), in %")
            plt.title("Detection Error Trade-off (DET) Curve")
            plt.savefig(
                f'picture/score/lpc_gmm/{speaker_to_identify}_det.png'
            )
            plt.close(fig)

            fig = plt.figure()
            plt.plot(far, 1 - np.array(frr))
            plt.xlabel("False Accept Rate(FAR), in %")
            plt.ylabel("Correct Accept Rate(CAR), in %")
            plt.title("Receiver Operation Characteristic (ROC) Curve")
            plt.savefig(
                f'picture/score/lpc_gmm/{speaker_to_identify}_roc.png'
            )
            plt.close(fig)

            plt.plot(far, 1 - np.array(frr))

            if (j + 1) % 10 == 0:
                plt.savefig(
                    f'picture/score/mfcc_gmm/global_{j // 10}_roc.png'
                )
                plt.close(fig)
                fig = plt.figure()


class SVMClassifier(Classifier):
    def __init__(self):
        self.svm = None
        self.shape = 9737

    @get_function_duration
    @get_function_memory_consumption
    def train(self, features: list, labels: list):
        """
        Trains SVM using the given features and corresponding labels

        :param features: (list) list of the features for each sample
        :param labels: (list) corresponding label for each sample
        :return:
        """

        reshape_features, _ = reshape_as_2d_array(features)

        self.svm = svm.SVC()
        self.svm.fit(reshape_features, labels)

    @get_function_duration
    @get_function_memory_consumption
    def classify(self, features: list, labels: list):
        """
        Generates results for the classifier SVM.
        'features' and 'labels' must have the same dimension

        :param features: (list) list of the features for each sample
        :param labels: (list) corresponding label for each sample
        """

        if not self.svm:
            print("The classifier must first be trained "
                  "before being able to classify")

        reshape_features, _ = reshape_as_2d_array(features)

        results = self.svm.predict(reshape_features)

        get_confusion_matrix(np.array(labels), results)

        return labels, results


if __name__ == "__main__":
    gmm = GMMClassifier()
    gmm.classify([2], [1])
