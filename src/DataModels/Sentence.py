class Sentence:
    def __init__(self, sentence, begin_idx, end_idx):
        self.sentence = sentence
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.list_entities = list()

    def add_entity(self, entity):
        self.list_entities.append(entity)
        pass

    def has_entity(self):
        if len(self.list_entities) ==0:
            return False
        return True

    def has_substance_abuse_entity(self):
        substance_abuse_entity_types = {"Alcohol", "Tobacco", "Drug"}
        for entity in self.list_entities:
            if entity.type in substance_abuse_entity_types:
                return True
        return False