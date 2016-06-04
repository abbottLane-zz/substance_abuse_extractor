from DataLoader import Globals as g
from DataLoader import Configuration as c
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from Classification import Classifier
from sklearn.svm import LinearSVC
import numpy as np


def fill_events(info, attrib_classifier, feature_map):
    sent_objs = info.sent_objs

    # Find events per sentence
    events_per_sent = [[] for _ in sent_objs]   # list[event_list] -- event_list= list[{}]
    for e in info.predicted_event_objs_by_index:
        events_per_sent[e] = info.predicted_event_objs_by_index[e]

    # Find attributes per sentence
    attribs_per_sent = [[] for _ in sent_objs]   # list[list[Attributes]]
    for a in info.tok_sent_with_crf_classification:
        attribs_per_sent[a] = info.tok_sent_with_crf_classification[a]

    # Stuff attribs into events
    fill(sent_objs, events_per_sent, attribs_per_sent, attrib_classifier, feature_map)


def fill(sent_objs, events_per_sent, attribs_per_sent, attrib_classifier, feature_map):
    for sent_obj, events, attribs in zip(sent_objs, events_per_sent, attribs_per_sent):
        # If there's just one event, stuff all attributes in
        if len(events) == 1:
            for attrib_type in attribs:
                events[0].attributes_list.append(attribs[attrib_type][0])

        # Else, ya gotta get more tricky
        else:
            assign_attribs_to_events(sent_obj, events, attribs, attrib_classifier, feature_map)


def assign_attribs_to_events(sent_obj, events, attribs, attrib_classifier, feature_map):
    for a in attribs:
        attrib = attribs[a]

        # Get data
        attrib_feats = __grab_attribute_feats(attrib, sent_obj, events)
        number_of_features = len(feature_map)

        # Vectorize sentences and classify
        test_vector = Classifier.vectorize_test_sent(attrib_feats, feature_map)
        test_array = np.reshape(test_vector, (1, number_of_features))
        classification = attrib_classifier.predict(test_array)

        # Assign attributes based on classifcations
        assigned_event = False
        for e in events:
            event = events[e]

            # If classified as this event
            if classification[0] == event.type:
                event.attributes_list.append(attrib)
                assigned_event = True

        # If assigned to existing substance type, put into one that is
        if not assigned_event:
            events[0].attributes_list.append(attrib)


def train_event_filler(training_doc_objs):
    feature_extractor = FeatureExtractor(training_doc_objs)

    # Get feature
    features, labels = __features_and_labels(feature_extractor)
    attrib_vectors, labels_for_classifier, feature_map = Classifier.vectorize_data(features, labels)

    # Create Model
    classifier = LinearSVC()
    classifier.fit(attrib_vectors, labels_for_classifier)

    return classifier, feature_map


def __features_and_labels(feature_extractor):
    feature_sets = []
    label_sets = []

    documents = feature_extractor.documents
    for key in documents:
        doc_sent_objs = documents[key].get_sentence_obj_list()

        for sent_obj in doc_sent_objs:
            attrib_features, gold_labels = __attribute_feats_and_labels(sent_obj)

            for feats, labels in zip(attrib_features, gold_labels):
                feature_sets.append(feats)
                label_sets.append(labels)

    return feature_sets, label_sets


def __attribute_feats_and_labels(sent_obj):
    attrib_feature_dicts = []
    labels = []

    events = [e for e in sent_obj.set_entities if (e.type in g.SPECIFIC_CLASSIFIER_TYPES)]

    for event in events:
        substance = event.type

        for attrib in event.dict_of_attribs:
            attrib_features = __grab_attribute_feats(event.dict_of_attribs[attrib], sent_obj, events)
            attrib_feature_dicts.append(attrib_features)
            labels.append(substance)

    return attrib_feature_dicts, labels


def __grab_attribute_feats(attrib, sent_obj, events):
    attrib_feature_dict = {}

    # All event types found in sentence - learn which one to assign to
    __add_events_in_sent(attrib_feature_dict, events)

    # Attribute type
    feat = g.ATTRIB_TYPE + attrib.type
    attrib_feature_dict[feat] = True

    # Attribute unigrams
    __add_attrib_words(attrib_feature_dict, attrib)

    # Surrounding words
    __add_surrounding_words(attrib_feature_dict, sent_obj, attrib)

    # Words between attrib and mention?

    return attrib_feature_dict


def __tokenize(sentence):
    tok_sent = [w.lower() for w in sentence.split()]
    # TODO -- same tokenization as the Attribute Extractor
    return tok_sent


def __add_events_in_sent(attrib_feature_dict, events):
    for event in events:
        feat = g.EVENT_TYPE + event.type
        attrib_feature_dict[feat] = True


def __add_attrib_words(attrib_feature_dict, attrib):
    grams = attrib.text.split()
    for gram in grams:
        feat = g.HAS_GRAM + gram
        attrib_feature_dict[feat] = True


def __add_surrounding_words(attrib_feature_dict, sent_obj, attrib):
    sentence = sent_obj.sentence

    # Tokenize surrounding words
    attrib_start_index = int(attrib.span_begin) - sent_obj.begin_idx
    after_attrib_index = attrib_start_index+len(attrib.text)
    pre_attrib = sentence[:attrib_start_index]
    post_attrib = sentence[after_attrib_index:]

    pre_tok = __tokenize(pre_attrib)
    post_tok = __tokenize(post_attrib)

    # Add unigrams within window
    window_start_index = len(pre_tok) - c.SURR_WORDS_WINDOW
    for pre in pre_tok[window_start_index:]:
        pre_feat = g.SURR_WORD + pre
        attrib_feature_dict[pre_feat] = True

    for post in post_tok[:c.SURR_WORDS_WINDOW]:
        post_feat = g.SURR_WORD + post
        attrib_feature_dict[post_feat] = True
