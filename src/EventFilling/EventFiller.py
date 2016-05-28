from DataModels import Sentence
from DataModels.Event import Event


def fill_events(sent_info, substance_types, statuses):
    sents = []              # list[bare sentences]
    events_per_sent = []    # list[event_list] -- event_list= list[{}]
    attribs_per_sent = []   # list[list[Attributes]]

    # If there's just one event, stuff all attributes in
    if len(substance_types) == 1 and len(statuses) == 1:
        pass

    # Else, ya gotta get more tricky
    else:
        pass

