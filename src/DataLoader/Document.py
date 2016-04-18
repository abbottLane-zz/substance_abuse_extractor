class Document:
    def __init__(self, id, list_of_sentences):
        self.id = id
        self.sentences = list_of_sentences

    def get_id(self):
        return self.id
    def get_sentences(self):
        return self.sentences