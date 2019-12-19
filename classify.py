#!/usr/bin/env python3
# coding: utf8

from sklearn import mixture


def train_gmm(features: list) -> mixture.GaussianMixture:
    """
    Trains gmm using the given features

    :param features: (list) of features
    :return: trained gmm
    """

    gmm = mixture.GaussianMixture(
        n_components=3, covariance_type='diag', n_init=3
    )
    gmm.fit(np.vstack(features))

    return gmm
