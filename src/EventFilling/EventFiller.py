from DataModels import Sentence
from DataModels.Event import Event


def fill_events(info):
    # Set up sents, events, attributes
    sents = []              # list[bare sentences]
    events_per_sent = [[] for _ in sents]   # list[event_list] -- event_list= list[{}]
    for e in info.predicted_event_objs_by_index:
        events_per_sent[e] = info.predicted_event_objs_by_index[e]

    attribs_per_sent = [[] for _ in sents]   # list[list[Attributes]]
    for a in info.tok_sent_with_crf_classification:
        attribs_per_sent[a] = info.tok_sent_with_crf_classification[a]

    # Stuff attribs into events
    fill(sents, events_per_sent, attribs_per_sent)


def fill(sents, events_per_sent, attribs_per_sent):
    for sent, events, attribs in zip(sents, events_per_sent, attribs_per_sent):
        # If there's just one event, stuff all attributes in
        if len(events) == 1:
            for attrib_type in attribs:
                events[0].attributes_list.append(attribs[attrib_type][0])

        # Else, ya gotta get more tricky
        else:
            pass
