import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import re
import string
from Classification import Globals
from Classification import SentInfo


def train_models(feature_extractor):
    classifiers = {}
    feat_maps = {}

    sent_obj_classes = __sentences_and_labels(feature_extractor)

    # General Substance Classifier
    subst_classifier, subst_feat_map = train_model(sent_obj_classes.processed_sents,
                                                   sent_obj_classes.get_labels(Globals.SUBSTANCE))
    classifiers[Globals.SUBSTANCE] = subst_classifier
    feat_maps[Globals.SUBSTANCE] = subst_feat_map

    # Specific classifiers
    for class_type in Globals.SPECIFIC_CLASSIFIER_TYPES:
        sent_objs, proc_sents, labels = sent_obj_classes.get_sentences_w_classf_type(class_type)
        classifier, feat_map = train_model(proc_sents, labels)
        classifiers[class_type] = classifier
        feat_maps[class_type] = feat_map

    return classifiers, feat_maps


def train_model(proc_sents, labels):
        # Convert Data to vectors
        sent_vectors, labels_for_classifier, feature_map = __vectorize_data(proc_sents, labels)

        # Create Model
        classifier = LinearSVC()
        classifier.fit(sent_vectors, labels_for_classifier)

        return classifier, feature_map


def get_classifications(classifiers, feat_maps, feat_extractor):

    sent_info = __sentences_and_labels(feat_extractor)

    # Classify Substance
    sent_info = classify(classifiers[Globals.SUBSTANCE], Globals.SUBSTANCE,
                            feat_maps[Globals.SUBSTANCE], sent_info)

    # Specific classifications
    for classf in classifiers:
        if classf != Globals.SUBSTANCE:
            sent_info = classify(classifiers[classf], classf, feat_maps[classf], sent_info)

    return sent_info


def classify(classifier, classifier_type, feature_map, sent_info):

    # Get data
    sent_indices, proc_sents, labels = sent_info.get_sentences_w_classf_type(classifier_type)

    # Temporary test sentences
    '''
    # TODO -- replace temp sents with real sents
    none_sent = "Patient \likes $3.25 baseball"
    fat = "Boy is this /patient 3.2% fat"
    all_sent = 'Patient is "100% a 1.3 non-smoker"'
    sucker = "Patient 'smokes', 8 packs*&& a day."
    test_sents = [none_sent, fat, all_sent, sucker]
    print("\nTest sentences:\n\t" + str(test_sents))
    '''

    number_of_sentences = len(proc_sents)
    number_of_features = len(feature_map)

    # Vectorize sentences and classify
    test_vectors = [__vectorize_test_sent(sent, feature_map) for sent in proc_sents]
    test_array = np.reshape(test_vectors, (number_of_sentences, number_of_features))
    classifications = classifier.predict(test_array)

    # Grab sents with substance info
    sents_w_info = []
    for index, classification in zip(sent_indices, classifications):
        if classification == Globals.HAS_SUBSTANCE:
            sents_w_info.append(index)

    sent_info.predicted_class_sent_lists[classifier_type] = sents_w_info

    return sent_info


def __sentences_and_labels(feature_extractor):
    all_sent_objs = []
    proc_sents = []

    gold_labels = {}
    gold_labels[Globals.SUBSTANCE] = set()
    gold_labels[Globals.ALCOHOL] = set()
    gold_labels[Globals.DRUGS] = set()
    gold_labels[Globals.TOBACCO] = set()

    sent_index = 0

    documents = feature_extractor.documents
    for key in documents:
        doc_sent_objs = documents[key].get_sentence_obj_list()
        all_sent_objs.append(doc_sent_objs)

        for sent_obj in doc_sent_objs:
            # Preprocess sentence
            sentence = {}
            grams = __process_sentence(sent_obj.sentence)

            # - Features -
            # Unigrams
            for gram in grams:
                sentence[gram] = True
            proc_sents.append(sentence)

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

    sent_classes = SentInfo.SentInfo(all_sent_objs, proc_sents, gold_labels)

    return sent_classes


def __vectorize_data(sentences, labels):
    # convert to vectors
    dict_vec = DictVectorizer()
    sentence_vectors = dict_vec.fit_transform(sentences).toarray()

    # create feature map
    feature_names = dict_vec.get_feature_names()
    feature_map = {}
    for index, feat in enumerate(feature_names):
        feature_map[feat] = index

    return sentence_vectors, np.array(labels), feature_map


def __vectorize_test_sent(sentence, feature_map):
    vector = [0 for _ in range(len(feature_map))]
    #grams = __process_sentence(sentence)
    grams = sentence.split()
    for gram in grams:
        if gram in feature_map:
            index = feature_map[gram]
            vector[index] = 1
    return vector


def __process_sentence(sentence):
    # TODO -- remove 'SOCIAL HISTORY:' and variants

    sentence = sentence.lower()
    grams = sentence.split()
    processed_grams = []

    left_omitted_chars = "|".join(["\$", "\."])
    right_omitted_chars = "|".join(["%"])
    ending_punc = re.sub(right_omitted_chars, "", string.punctuation)
    starting_punc = re.sub(left_omitted_chars, "", string.punctuation)

    for gram in grams:
        # Remove punctuation
        gram = gram.rstrip(ending_punc)
        gram = gram.lstrip(starting_punc)

        # TODO -- prune unuseful words

        if gram:
            # Compress into word classes
            if gram.isdigit():
                processed_grams.append(Globals.NUMBER)
            elif re.sub("\.", "", gram).isdigit():
                processed_grams.append(Globals.DECIMAL)
            elif gram[0] == '$':
                processed_grams.append(Globals.MONEY)
            elif gram[len(gram)-1] == '%':
                processed_grams.append(Globals.PERCENT)
            else:
                processed_grams.append(gram)

    return processed_grams
