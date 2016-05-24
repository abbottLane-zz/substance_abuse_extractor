from DataLoader import Globals


class Sentence:
    def __init__(self, sentence, begin_idx, end_idx):
        self.sentence = sentence
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.set_entities = set()
        self.labeled_type = None

    def add_entity(self, entity):
        self.set_entities.add(entity)
        pass

    def get_status_label_and_evidence(self, type):
        for ent in self.set_entities:
            if ent.type == type:
                for attrib in ent.dict_of_attribs.values():
                    if attrib.type == "Status":
                        return attrib.a_attrib.status, attrib.text
        return "unknown", "evidence unavailable"

    def get_event_by_type(self, type):
        for event in self.set_entities:
            if event.type == type:
                return event
        return None

    def has_entity(self):
        if len(self.set_entities) == 0:
            return False
        return True

    def has_substance_abuse_entity(self):
        for entity in self.set_entities:
            if entity.type in Globals.SPECIFIC_CLASSIFIER_TYPES:
                return True
        return False

    def set_labeled_type(self, type):
        self.labeled_type = type

    def has_specific_abuse_entity(self, classification_type):
        for entity in self.set_entities:
            if entity.type == classification_type:
                return True
        return False
