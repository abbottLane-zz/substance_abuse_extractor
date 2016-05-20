from DataLoader import Globals


class Entity:
    def __init__(self, tag, dict_of_attribs, type):
        self.tag = tag
        self.dict_of_attribs = dict_of_attribs
        self.type = type

    def get_entity_begin_idxs(self):
        idxs = list()
        for attrib_key in self.dict_of_attribs:
            idxs.append(self.dict_of_attribs[attrib_key].span_begin)
        return idxs

    def is_substance_abuse(self):
        if self.type in Globals.SPECIFIC_CLASSIFIER_TYPES:
            return True
        else:
            return False
