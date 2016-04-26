class Attrib:
    def __init__(self, tag, type, span_begin, span_end, text):
        self.type = type
        self.tag = tag
        self.span_begin = span_begin
        self.span_end = span_end
        self.text = text