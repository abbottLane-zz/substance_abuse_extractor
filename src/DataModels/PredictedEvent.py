class PredictedEvent:
    def __init__(self, type, status, sent_id,idx):
        self.type = type
        self.status = status
        self.sentence = sent_id
        self.sent_obj_idx=idx
        self.attributes_list=list()

    def add_event_attribute(self, attrib):
        self.attributes_list.append(attrib)
