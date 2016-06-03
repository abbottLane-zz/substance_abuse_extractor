from DataModels import Sentence
from DataModels.Event import Event
from DataLoader import Globals as g
from DataLoader import Configuration as c
from FeatureExtractor.FeatureExtractor import FeatureExtractor


def fill_events(info, attrib_classifier):
    # Set up sents, events, attributes
    tok_sents = []              # list[bare sentences]
    events_per_sent = [[] for _ in tok_sents]   # list[event_list] -- event_list= list[{}]
    for e in info.predicted_event_objs_by_index:
        events_per_sent[e] = info.predicted_event_objs_by_index[e]

    attribs_per_sent = [[] for _ in tok_sents]   # list[list[Attributes]]
    for a in info.tok_sent_with_crf_classification:
        attribs_per_sent[a] = info.tok_sent_with_crf_classification[a]

    # Stuff attribs into events
    fill(tok_sents, events_per_sent, attribs_per_sent, attrib_classifier)


def fill(tok_sents, events_per_sent, attribs_per_sent, attrib_classifier):
    for sent, events, attribs in zip(tok_sents, events_per_sent, attribs_per_sent):
        # If there's just one event, stuff all attributes in
        if len(events) == 1:
            for attrib_type in attribs:
                events[0].attributes_list.append(attribs[attrib_type][0])

        # Else, ya gotta get more tricky
        else:
            assign_attribs_to_events(sent, events, attribs, attrib_classifier)


def assign_attribs_to_events(sent, events, attribs, attrib_classifier):
    pass


def train_event_filler(training_doc_objs):
    feature_extractor = FeatureExtractor(training_doc_objs)

    features, labels = __features_and_labels(feature_extractor)
    pass


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
                label_sets.append(gold_labels)

    return feature_sets, label_sets


def __attribute_feats_and_labels(sent_obj):
    attrib_feature_dicts = []
    labels = []

    events = [e for e in sent_obj.set_entities if (e.type in g.SPECIFIC_CLASSIFIER_TYPES)]

    for event in events:
        substance = event.type

        for attrib in event.dict_of_attribs:
            attrib_features = __grab_attribute_feats(attrib, sent_obj, events)
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
    attrib_start_index = attrib.span_begin - sent_obj.begin_idx
    after_attrib_index = attrib_start_index+len(attrib.text)
    pre_attrib = sentence[:attrib_start_index]
    post_attrib = sentence[after_attrib_index:]

    pre_tok = __tokenize(pre_attrib)
    post_tok = __tokenize(post_attrib)

    # Add unigrams within window
    for pre, post in zip(pre_tok, post_tok)[:c.SURR_WORDS_WINDOW]:
        pre_feat = g.SURR_WORD + pre
        post_feat = g.SURR_WORD + post
        attrib_feature_dict[pre_feat] = True
        attrib_feature_dict[post_feat] = True
