from DataModels.Entity import Entity
from DataModels.TAttrib import TAttrib
from DataModels.AAttrib import AAttrib

class AnnotationDoc:
    def __init__(self, id, list_of_annotations):
        self.id = id
        self.line_text_list = list_of_annotations

        # Break down into Entity Objects
        self.list_of_entities = self.process_annotation_lines(self.line_text_list)

    def get_id(self):
        return self.id
    def get_entities_as_list(self):
        return self.list_of_entities
    def get_text_lines(self):
        return self.line_text_list

    def process_annotation_lines(self, list_of_ann_lines):
        # On a per document basis, process a list of annotations:
        #     First pass through data:
        #       create tuples for each line, store in a list for the second pass through data
        #       create Entity objects for all 'E' tags
        #       store all other lines in the attributes dict by handle, ie "T5" or "A3"
        entity_list = list()
        attrib_dict = dict()
        entity_obj_list = list()

        for line in list_of_ann_lines:
            line_tokens = line.split()
            tokens_tuple = tuple(line_tokens)
            if 'E' in tokens_tuple[0]:
                entity_list.append(tokens_tuple)
            else:
                attrib_dict[tokens_tuple[0]] = tokens_tuple

        # foreach entity, construct its object
        for entity in entity_list:
            Etag = entity[0]
            entity_attrib_dict = dict()
            for attrib in entity[1:]:
                attrib_obj = self.create_entity_attribute_object(attrib, attrib_dict)
                entity_attrib_dict[attrib_obj.tag] = attrib_obj

            entity_obj_list.append(Entity(Etag, entity_attrib_dict))
        return entity_obj_list

    def create_entity_attribute_object(self, attrib, attrib_dict):
        type_tag_pair = attrib.split(':')
        real_attrib = attrib_dict[type_tag_pair[1]] # Retrieve the T attribute by pulling its tag from attrib dict
        # real_attribs look like this: Tuple('T1','Tobacco','38','43','smoker')
        #   ie: Tuple(tag, type, span_begin, span_end, 'possibly', 'multiple', 'values', 'of', 'text')
        real_tag = real_attrib[0]
        real_type = real_attrib[1]
        real_span_begin = real_attrib[2]
        real_span_end = real_attrib[3]
        real_text = ""
        for index, item in enumerate(real_attrib[4:], start=4):
            real_text = real_text + " " + item

        #check for the presence of an A attribute. There will not always be one
        for val in attrib_dict.keys():
            if 'A' in val:
                if attrib_dict[val][2] == real_tag:
                    return TAttrib(real_tag, real_type, real_span_begin, real_span_end, real_text.lstrip(), self.make_a_type_attrib(attrib_dict[val]))
        return TAttrib(real_tag, real_type, real_span_begin, real_span_end, real_text.lstrip(), None)

    def make_a_type_attrib(self, a_type_tuple):
        tag = a_type_tuple[0]
        pointer = a_type_tuple[2]
        status = a_type_tuple[3]

        return AAttrib(tag, pointer, status)


