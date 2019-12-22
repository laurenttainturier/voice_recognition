#!/usr/bin/env python3
# coding: utf8

import sys
from time import time

import numpy as np


def get_function_duration(old_function):
    """
    The role of this decorator is to evaluate
    the execution time of a function

    :param old_function: (Function) to evaluate
    :return: new_function (Function)
    """
    def new_function(*args, **kwargs):
        # print(f"{old_function.__name__}", end=" ")
        starting_time = time()
        result = old_function(*args, **kwargs)
        print(f"  - duration: {time() - starting_time}s")

        return result

    return new_function


def get_function_memory_consumption(old_function):
    """
    The role of this decorator is to evaluate
    the memory consumption of a function

    :param old_function: (Function) to evaluate
    :return: new_function (Function)
    """
    def new_function(*args, **kwargs):
        # print(f"{old_function.__name__}", end=" ")
        starting_memory = sys.getallocatedblocks()
        result = old_function(*args, **kwargs)
        print(
            f"  - memory: "
            f"{sys.getallocatedblocks() - starting_memory} blocs"
        )

        return result

    return new_function


def reshape_as_2d_array(
        features: list, shape_min: int = None) -> np.array:
    """
    Reshape a list of 2d arrays with note the same dimension

    :param features: (list) to be reshaped
    :param shape_min:
    :return: (np.array)
    """
    shape = []
    flatten_features = []

    for x in features:
        shape.append(x.shape[0] * x.shape[1])
        flatten_features.append(x.flatten())

    if not shape_min:
        shape_min = min(shape)

    return np.array([
        feature[:shape_min]
        for feature in flatten_features
    ]), shape_min


def get_confusion_matrix(
        expected_labels: np.array,
        predicted_labels: np.array) -> np.array:
    """
    Returns the confusion matrix

    :param expected_labels: (np.array)
    :param predicted_labels: (np.array)
    :return: (np.array)
    """

    unique_labels = np.unique(expected_labels)
    label_nb = len(unique_labels)
    confusion_matrix = np.zeros((label_nb, label_nb))
    for m_pos, label in enumerate(unique_labels):
        indices = np.where(expected_labels == label)[0]
        for i in indices:
            if predicted_labels[i] == m_pos:
                confusion_matrix[m_pos, m_pos] += 1
            else:
                error_pos = unique_labels.tolist().index(
                    predicted_labels[i])
                confusion_matrix[m_pos, error_pos] += 1
        # confusion_matrix[m_pos] //= len(indices)

    print(confusion_matrix)

    return confusion_matrix
