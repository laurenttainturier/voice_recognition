#!/usr/bin/env python3
# coding: utf8

from segment import *
from extract import *
from classify import *

from visualize import display_proportions


# extraction method
MFCC = ("MFCC", extract_with_mfcc)
LPC = ("LPC", extract_with_lpc)
PLP = ("PLP", extract_with_plp)

# classification method
GMM = ("GMM", GMMClassifier())
SVM = ("SVM", SVMClassifier())

EXTRACTOR = (MFCC, LPC, PLP)
CLASSIFIER = (GMM, SVM)


if __name__ == "__main__":
    extractor_name, extractor = MFCC
    classifier_name, classifier = GMM

    print(
        f"Classification with {classifier_name} "
        f"using extractor {extractor_name}"
    )

    print("-> segmentation starts")
    original_dataset = create_dataframe('database/dev/keys/meta.lst')

    cls_dataset = extract_dataset(original_dataset)
    ubm_dataset = extract_background_dataset(original_dataset)

    display_proportions(
        pd.concat([cls_dataset, ubm_dataset]),
        [GENDER, MICRO, DEGRADATION, SPEAKER_NUMBER]
    )

    cls_train_dataset, cls_test_dataset = split_dataset(cls_dataset, 30)

    print("-> feature extraction starts")
    cls_train_features, cls_train_labels = extract_features(
        cls_train_dataset, extractor, multi=False
    )
    cls_test_features, cls_test_labels = extract_features(
        cls_test_dataset, extractor, multi=False
    )

    ubm_features, _ = extract_features(
        ubm_dataset, extractor, multi=False
    )

    print("-> training starts")
    classifier.train(cls_train_features, cls_train_labels)

    print("-> training background")
    classifier.train_background(ubm_features)

    print("-> classifying starts")
    classifier.classify(cls_test_features, cls_test_labels)
