from DataLoader import Globals as g
from DataLoader import Configuration as c
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from DataModels.TAttrib import TAttrib
from Classification import Classifier
from sklearn.svm import LinearSVC
import numpy as np
import re


def fill_events(info, attrib_classifier, feature_map):
    sent_objs = info.sent_objs

    # Find events per sentence
    events_per_sent = [[] for _ in sent_objs]   # list[event_list] -- event_list= list[{}]
    for e in info.predicted_event_objs_by_index:
        events_per_sent[e] = info.predicted_event_objs_by_index[e]

    # Find attributes per sentence
    attribs_per_sent = [get_attribs_from_sentence(s) for s in sent_objs]   # list[list[Attributes]]

    # Stuff attribs into events
    fill(sent_objs, events_per_sent, attribs_per_sent, attrib_classifier, feature_map)


def get_attribs_from_sentence(sent_obj):
    attributes = []
    attribs_gram_lists = []
    start_indicies = []
    end_indicies = []
    types = []

    sent_attrib_predictions = sent_obj.tok_sent_with_crf_predicted_attribs
    for attrib_type in sent_attrib_predictions:
        in_attrib = False

        for index, tuple in enumerate(sent_attrib_predictions[attrib_type]):
            gram = tuple[0]
            prediction = tuple[1]
            span_start = tuple[2]
            span_end = tuple[3]

            if prediction == attrib_type:
                # Create new attrib if at the beginning of a new attrib
                if not in_attrib:
                    attribs_gram_lists.append([])
                    start_indicies.append(span_start)
                    end_indicies.append(span_end)
                    types.append(attrib_type)
                    in_attrib = True
                else:
                    # Keep updating the end index
                    end_indicies[-1] = span_end

                # Add to current attrib until attrib label changes
                attribs_gram_lists[-1].append(gram)
            else:
                in_attrib = False

    for tag_index, (gram_list, start_index, end_index, type) in enumerate(zip(attribs_gram_lists, start_indicies, end_indicies, types)):
        tag = "T" + str(tag_index)
        text, end_index = recover_text_from_span(sent_obj, start_index, end_index)
        attrib = TAttrib(tag, type, start_index, end_index, text, None)
        attributes.append(attrib)

    return attributes


def fill(sent_objs, events_per_sent, attribs_per_sent, attrib_classifier, feature_map):
    for sent_obj, events, attribs in zip(sent_objs, events_per_sent, attribs_per_sent):
        # If there's just one event, stuff all attributes in
        if len(events) == 1:
            for attrib_type in attribs:
                events[0].attributes_list.append(attribs[attrib_type][0])

        # Else, get more tricky
        else:
            assign_attribs_to_events(sent_obj, events, attribs, attrib_classifier, feature_map)

        # Add statuses to statuses to State


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

            # If classified as this event type
            if classification[0] == event.type:
                event.attributes_list.append(attrib)
                assigned_event = True

        # If assigned to existing substance type, put into one that is
        if not assigned_event:
            events[0].attributes_list.append(attrib)


def recover_text_from_span(sent_obj, start_index, end_index):
    start = start_index - sent_obj.begin_idx
    end = end_index - sent_obj.begin_idx
    text = sent_obj.sentence[start:end]

    # Strip off any ending punctuation
    has_ending_punc = re.match(r"(.*)[\.,?!]$", text)
    if has_ending_punc:
        text = has_ending_punc.group(1)
        end_index -= 1

    return text, end_index


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
