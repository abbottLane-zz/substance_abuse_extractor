from DataModels import Sentence
from EventCreator.Event import Event


def create_events(sent_obj: Sentence, substance_types, statuses):
    events = []

    # If there's just one event, stuff all attributes in
    if len(substance_types) == 1 and len(statuses) == 1:
        event = Event(substance_types[0], statuses[0], sent_obj.set_attributes, sent_obj.sentence)
        events.append(event)

    # Else, ya gotta get more tricky
    else:
        pass

    return events

