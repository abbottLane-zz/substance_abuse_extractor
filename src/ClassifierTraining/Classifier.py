import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import re
import string

HAS_SUBSTANCE = "has_subs_info"
NO_SUBSTANCE = "no_subs_info"


def train_model(feature_extractor):
        # Convert Data to vectors
        orig_sents, sents, labels = __sentences_and_labels(feature_extractor)
        sent_vectors, labels, feature_map = __vectorize_data(sents, labels)

        # Create Model
        classifier = LinearSVC()
        classifier.fit(sent_vectors, labels)

        return classifier, feature_map


def classify_sentences(classifier, feature_map, feat_extractor):

    # Get data
    # orig_sents, proc_sents, labels = __sentences_and_labels(feat_extractor)

    # Temporary test sentences
    # TODO -- replace temp sents with real sents
    none_sent = "Patient \likes $3.25 baseball"
    fat = "Boy is this /patient 3.2% fat"
    all_sent = 'Patient is "100% a 1.3 non-smoker"'
    sucker = "Patient 'smokes', 8 packs*&& a day."
    test_sents = [none_sent, fat, all_sent, sucker]
    print("\nTest sentences:\n\t" + str(test_sents))

    number_of_sentences = len(test_sents)
    number_of_features = len(feature_map)

    # Vectorize sentences and classify
    proc_sents = [" ".join(__process_sentence(sent)) for sent in test_sents]
    test_vectors = [__vectorize_test_sent(sent, feature_map) for sent in proc_sents]
    test_array = np.reshape(test_vectors, (number_of_sentences, number_of_features))
    classifications = classifier.predict(test_array)

    # Grab sents with substance info
    orig_sents_w_info = []
    proc_sents_w_info = []
    for orig_sent, proc_sent, classification in zip(test_sents, proc_sents, classifications):
        if classification == HAS_SUBSTANCE:
            orig_sents_w_info.append(orig_sent)
            proc_sents_w_info.append(proc_sent)

    return orig_sents_w_info, proc_sents_w_info


def __sentences_and_labels(feature_extractor):
    orig_sents = []
    proc_sents = []
    labels = []

    documents = feature_extractor.documents
    for key in documents:
        for sent_obj in documents[key].get_sentence_obj_list():
            # Find the words contained to be used as unigram features
            sentence = {}
            orig_sent = sent_obj.sentence
            orig_sents.append(orig_sent)
            grams = __process_sentence(orig_sent)

            # TODO -- add useful features
            for gram in grams:
                sentence[gram] = True
            proc_sents.append(sentence)

            # Track gold labels
            if sent_obj.has_substance_abuse_entity():
                labels.append(HAS_SUBSTANCE)
            else:
                labels.append(NO_SUBSTANCE)

    return orig_sents, proc_sents, labels


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
    NUMBER = "NUMBER"
    DECIMAL = "DECIMAL"
    MONEY = "MONEY"
    PERCENT = "PERCENT"

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
                processed_grams.append(NUMBER)
            elif re.sub("\.", "", gram).isdigit():
                processed_grams.append(DECIMAL)
            elif gram[0] == '$':
                processed_grams.append(MONEY)
            elif gram[len(gram)-1] == '%':
                processed_grams.append(PERCENT)
            else:
                processed_grams.append(gram)

    return processed_grams
