import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from Classification import Globals
from Classification import SentInfo
from Classification.Classifier import __sentences_and_labels, __vectorize_data, __vectorize_test_sent, \
    __process_sentence
from FeatureExtractor.FeatureExtractor import FeatureExtractor


def train_status_classifiers(sent_info):
    classifiers = {}
    feat_maps = {}
    feat_dicts ={}

    alcohol_sents = sent_info.get_sentences_w_info(Globals.ALCOHOL)
    drug_sents = sent_info.get_sentences_w_info(Globals.DRUGS)
    tobac_sents = sent_info.get_sentences_w_info(Globals.TOBACCO)

    #Label each sentence with its type
    for sent in alcohol_sents:
        sent.set_labeled_type(Globals.ALCOHOL)
    for sent in drug_sents:
        sent.set_labeled_type(Globals.DRUGS)
    for sent in tobac_sents:
        sent.set_labeled_type(Globals.TOBACCO)

    # Create Feature-Label pairs for each Subs Abuse type
    alc_feats, alc_labels = get_features(alcohol_sents)
    drg_feats, drg_labels = get_features(drug_sents)
    tbc_feats, tbc_labels = get_features(tobac_sents)

    feat_dicts["Alcohol"] =alc_feats
    feat_dicts["Drug"] = drg_feats
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

def get_features(sents):
    feature_vecs = list()
    labels = list()
    for sent in sents:

        vector = dict()
        label,evidence =sent.get_status_label_and_evidence(sent.labeled_type)
        input_list = sent.sentence.lower().rstrip(",.!?:;").split()
        some_bigrams = list(get_bigrams(input_list))

        for pair in some_bigrams:
            vector[pair[0] + "_" + pair[1]] = True
        for x in input_list:
            vector[x]=True

        # This don't work, test data needs access to any features you choose
        #vector["evidence:"+evidence.lower()]=True

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


def __get_features_and_labels(feature_extractor):
    sent_objs = []
    orig_sents = []
    proc_sents = []
    sent_pre_vectors = []

    gold_labels = {}
    gold_labels[Globals.SUBSTANCE] = set()
    gold_labels[Globals.ALCOHOL] = set()
    gold_labels[Globals.DRUGS] = set()
    gold_labels[Globals.TOBACCO] = set()

    sent_index = 0

    documents = feature_extractor.documents
    for key in documents:
        doc_sent_objs = documents[key].get_sentence_obj_list()
        sent_objs.extend(doc_sent_objs)

        for sent_obj in doc_sent_objs:
            # Preprocess sentence
            sent_pre_vector = {}
            sentence = sent_obj.sentence

            orig_sents.append(sentence)
            grams = __process_sentence(sent_obj.sentence)
            proc_sents.append(" ".join(grams))

            # - Track sentences gold labelled as having substance -
            # Substance
            if sent_obj.has_substance_abuse_entity():
                gold_labels[Globals.SUBSTANCE].add(sent_index)

                # Alcohol
                if sent_obj.has_specific_abuse_entity(Globals.ALCOHOL):
                    gold_labels[Globals.ALCOHOL].add(sent_index)
                # Drugs
                if sent_obj.has_specific_abuse_entity(Globals.DRUGS):
                    gold_labels[Globals.DRUGS].add(sent_index)
                # Tobacco
                if sent_obj.has_specific_abuse_entity(Globals.TOBACCO):
                    gold_labels[Globals.TOBACCO].add(sent_index)

            sent_index += 1

    sent_pre_vectors, gold_labels = get_features(sent_objs)
    sent_info = SentInfo.SentInfo(sent_objs, orig_sents, proc_sents, sent_pre_vectors, gold_labels)
    return sent_info


def get_classifications(status_classifiers, status_feat_maps, feats_dicts, testing_feat_extractor):
    # TODO: collect features from testing data, and other data useful for our sentence object
    # not like this: sent_info = __sentences_and_labels(testing_feat_extractor)

    sent_info = __get_features_and_labels(testing_feat_extractor)


    # Specific classifications
    for classf in status_classifiers:
        sent_info = classify(status_classifiers[classf], classf, status_feat_maps[classf], sent_info, feats_dicts)

    return sent_info

def classify(classifier, classifier_type, feature_map, sent_info, feats_dicts):


    features = feats_dicts[classifier_type]
    number_of_sentences = len(features)
    number_of_features = len(feature_map)

    # Vectorize sentences and classify
    test_vectors = [__vectorize_test_sent(feats, feature_map) for feats in features]
    test_array = np.reshape(test_vectors, (number_of_sentences, number_of_features))
    classifications = classifier.predict(test_array)

    sent_info.predicted_classf_sent_lists[classifier_type] = classifications

    return sent_info