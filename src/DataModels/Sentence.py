class Sentence:
    def __init__(self, sentence, begin_idx, end_idx):
        self.sentence = sentence
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.list_entities = list()

    def add_entity(self, entity):
        self.list_entities.append(entity)
        pass