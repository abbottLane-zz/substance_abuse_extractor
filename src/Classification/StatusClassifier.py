import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from Classification import Globals
from Classification.Classifier import __sentences_and_labels, __vectorize_data, __vectorize_test_sent
from FeatureExtractor.FeatureExtractor import FeatureExtractor


def train_status_classifiers(sent_info):
    classifiers = {}
    feat_maps = {}
    feat_dicts ={}

    alcohol_sents = sent_info.get_sentences_w_info(Globals.ALCOHOL)
    drug_sents = sent_info.get_sentences_w_info(Globals.DRUGS)
    tobac_sents = sent_info.get_sentences_w_info(Globals.TOBACCO)

    # Create Feature-Label pairs for each Subs Abuse type
    alc_feats, alc_labels = get_features(alcohol_sents, Globals.ALCOHOL)
    drg_feats, drg_labels = get_features(drug_sents, Globals.DRUGS)
    tbc_feats, tbc_labels = get_features(tobac_sents, Globals.TOBACCO)

    feat_dicts["Alcohol"] =alc_feats
    feat_dicts["Drugs"] = drg_feats
    feat_dicts["Tobacco"]=tbc_feats

    # Create Model
    alc_classifier, alc_feature_map = train_model(alc_feats, alc_labels)
    drg_classifier, drg_feature_map = train_model(drg_feats, drg_labels)
    tbc_classifier, tbc_feature_map = train_model(tbc_feats, tbc_labels)

    # Set classifier in dictionary
    classifiers[Globals.ALCOHOL] = alc_classifier
    classifiers[Globals.DRUGS] = drg_classifier
    classifiers[Globals.TOBACCO] = tbc_classifier

    # Set feature map in dictionary
    feat_maps[Globals.ALCOHOL] = alc_feature_map
    feat_maps[Globals.DRUGS] = drg_feature_map
    feat_maps[Globals.TOBACCO] = tbc_feature_map

    return classifiers, feat_maps, feat_dicts

def get_features(sents, type):
    feature_vecs = list()
    labels = list()
    for sent in sents:

        vector = dict()
        label,evidence =sent.get_status_label_and_evidence(type)
        input_list = sent.sentence.lower().rstrip(",.!?:;").split()
        some_bigrams = list(get_bigrams(input_list))

        for pair in some_bigrams:
            vector[pair[0] + "_" + pair[1]] = True
        for x in input_list:
            vector[x]=True

        vector["evidence:"+evidence.lower()]=True

        feature_vecs.append(vector)
        labels.append(label)
    return feature_vecs, labels

def get_bigrams(input_list):
    return zip(input_list, input_list[1:])

def train_model(proc_sents, labels):
        # Convert Data to vectors
        sent_vectors, labels_for_classifier, feature_map = __vectorize_data(proc_sents, labels)

        # Create Model
        classifier = LinearSVC()
        classifier.fit(sent_vectors, labels_for_classifier)

        return classifier, feature_map


def get_classifications(status_classifiers, status_feat_maps, feats_dicts, testing_feat_extractor):
    sent_info = __sentences_and_labels(testing_feat_extractor)

    # Specific classifications
    for classf in status_classifiers:
        sent_info = classify(status_classifiers[classf], classf, status_feat_maps[classf], sent_info, feats_dicts)

    return sent_info

def classify(classifier, classifier_type, feature_map, sent_info, feats_dicts):

    for status_type in feats_dicts:
        features = feats_dicts[status_type]
        number_of_sentences = len(features)
        number_of_features = len(feature_map)

        # Vectorize sentences and classify
        test_vectors = [__vectorize_test_sent(feats, feature_map) for feats in features]
        test_array = np.reshape(test_vectors, (number_of_sentences, number_of_features))
        classifications = classifier.predict(test_array)



    return sent_info