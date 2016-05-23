from Classification import Globals

class Event:
    def __init__(self, tag, dict_of_attribs, type):
        self.tag = tag
        self.dict_of_attribs = dict_of_attribs
        self.type = type
        # self.predictedType=None
        # self.predictedStatus=None

    def get_entity_begin_idxs(self):
        idxs = list()
        for attrib_key in self.dict_of_attribs:
            idxs.append(self.dict_of_attribs[attrib_key].span_begin)
        return idxs
    def get_status(self):
        for attrib in self.dict_of_attribs.values():
            if attrib.type == "Status":
                return attrib.a_attrib.status
        return "unknown"

    def is_substance_abuse(self):
        if self.type in Globals.SPECIFIC_CLASSIFIER_TYPES:
            return True
        else:
            return False

    # def set_predicted_type(self, type):
    #     self.predictedType = type
    #
    # def set_predicted_status(self, status):
    #     self.predictedStatus = status
    #
    # def get_predicted_status(self):
    #     return self.predictedStatus
    # def get_predicted_type(self):
    #     return self.predictedType
