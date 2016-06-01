from DataModels import Sentence
from DataModels.Event import Event
from DataLoader import Globals as g
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
            attrib_features, gold_labels = attribute_feats_and_labels(sent_obj)

            for feats, labels in zip(attrib_features, gold_labels):
                feature_sets.append(feats)
                label_sets.append(gold_labels)

    return feature_sets, label_sets


def attribute_feats_and_labels(sent_obj):
    attrib_feature_dicts = []
    labels = []

    events = [e for e in sent_obj.set_entities if (e.type in g.SPECIFIC_CLASSIFIER_TYPES)]

    for event in events:
        substance = event.type

        for attrib in event.dict_of_attribs:
            attrib_features = grab_attribute_feats(attrib, sent_obj.sentence, events)
            attrib_feature_dicts.append(attrib_features)
            labels.append(substance)

    return attrib_feature_dicts, labels


def grab_attribute_feats(attrib, sentence, events):
    attrib_feature_dict = {}

    # All event types found in sentence - learn which one to assign to
    for event in events:
        feat = g.EVENT_TYPE + event.type
        attrib_feature_dict[feat] = True

    # Attribute type
    feat = g.ATTRIB_TYPE + attrib.type
    attrib_feature_dict[feat] = True

    # Attribute unigrams
    grams = attrib.text.split()
    for gram in grams:
        feat = g.HAS_GRAM + gram
        attrib_feature_dict[feat] = True

    # Surrounding words

    # Words between attrib and mention?

    # TODO -- get features from tokenized sentence
    tok_sent = tokenize_sentence(sentence)

    return attrib_feature_dict


def tokenize_sentence(sentence):
    tok_sent = []
    # TODO -- same tokenization as the Attribute Extractor
    return tok_sent
