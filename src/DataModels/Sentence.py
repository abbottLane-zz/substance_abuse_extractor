from Classification import Globals

class Sentence:
    def __init__(self, sentence, begin_idx, end_idx):
        self.sentence = sentence
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.set_entities = set()

    def add_entity(self, entity):
        self.set_entities.add(entity)
        pass

    def has_entity(self):
        if len(self.set_entities) == 0:
            return False
        return True

    def has_substance_abuse_entity(self):
        #substance_abuse_entity_types = {Globals.ALCOHOL, Globals.TOBACCO, Globals.DRUGS}
        for entity in self.set_entities:
            if entity.type in Globals.SPECIFIC_CLASSIFIER_TYPES:
                return True
        return False

    def has_specific_abuse_entity(self, classification_type):
        for entity in self.set_entities:
            if entity.type == classification_type:
                return True
        return False
