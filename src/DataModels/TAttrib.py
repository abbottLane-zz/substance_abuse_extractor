from DataLoader import Globals
from DataModels.AAttrib import AAttrib


class TAttrib:
    def __init__(self, tag, type, span_begin, span_end, text, a_attrib):
        self.type = type
        self.tag = tag
        self.span_begin = span_begin
        self.span_end = span_end
        self.text = text
        self.a_attrib = a_attrib

        if self.type =="Status" and a_attrib == None:
            self.a_attrib = AAttrib(None, None, "unknown")

