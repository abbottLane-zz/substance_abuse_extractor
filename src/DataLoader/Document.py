from nltk.tokenize import sent_tokenize


class Document:
    def __init__(self, id, list_of_paragraphs):
        self.id = id
        self.sentences_text_list = self.sent_segmentizer(list_of_paragraphs) #derive sentence list by segmentizing doc sentences
        self.original_text = self.rebuild_original_text() # Get original text so that spans are still accurate
        self.annotations = None
        self.sentence_obj_list = self.create_sentence_objs(self.sentences_text_list)

    def create_sentence_objs(self, sentences_text_list):

        pass

    def set_annotation(self, annotations):
        self.annotations = annotations

    def rebuild_original_text(self):
        full_doc =""
        iter_count =0
        for sent in self.sentences_text_list:
            if iter_count !=0:
                full_doc = full_doc + " " + sent
            else:
                full_doc = full_doc + sent
            iter_count = iter_count +1
        return full_doc

    def sent_segmentizer(self, sents):
        sentences = []
        for paragraph in sents:
            paragraph_sentences = sent_tokenize(paragraph)
            sentences.extend(paragraph_sentences)
        return sentences

    def get_id(self):
        return self.id
    def get_sentences_text_as_list(self):
        return self.sentences_text_list
    def get_original_text(self):
        return self.original_text