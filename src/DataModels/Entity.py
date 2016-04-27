class Entity:
    def __init__(self, tag, dict_of_attribs):
        self.tag = tag
        self.dict_of_attribs = dict_of_attribs

    def get_entity_sample_idx(self):
        for attrib_key in self.dict_of_attribs:
            return self.dict_of_attribs[attrib_key].span_begin
        return None