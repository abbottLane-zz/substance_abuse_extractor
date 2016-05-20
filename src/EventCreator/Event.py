from DataModels import Sentence


class Event:
    def __init__(self, substance, status, attributes, sentence):
        self.substance = substance
        self.status = status
        self.temporal = None
        self.method = None
        self.type = None
        self.amount = None
        self.frequency = None
        self.history = None

