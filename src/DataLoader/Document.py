from nltk.tokenize import sent_tokenize
from DataModels.Sentence import Sentence


class Document:
    def __init__(self, id, list_of_paragraphs):
        self.id = id
        self.sentences_text_list = self.sent_segmentizer(list_of_paragraphs) #derive sentence list by segmentizing doc sentences
        self.original_text = self.rebuild_original_text() # Get original text so that spans are still accurate
        self.annotations = None
        self.sentence_obj_list = self.create_sentence_objs(self.sentences_text_list)

    def create_sentence_objs(self, sentences_text_list):
        sent_objs = list()
        current_length = 0

        for sentence in sentences_text_list:
            if sentence == "\n":
                current_length = current_length +1
            if sentence != "\n":

                #debug
                # if "According to the chart," in sentence:
                #     holdit = 9

                sentence = sentence.rstrip()
                start_idx = current_length
                end_idx = current_length + len(sentence)
                sent_objs.append(Sentence(sentence, start_idx, end_idx))
                current_length = end_idx +1
        # Annotations have to be added to sentence objs. Use the set_annotations method below to do that
        return sent_objs

    def set_annotation(self, annotations):
        self.annotations = annotations
        #match entity spans to their sentences
        entities = self.annotations.get_entities_as_list()
        for entity in entities:
            entity_assigned = False
            for sent in self.sentence_obj_list:
                # if entity_assigned == True:
                #     entity_assigned=False
                #     break

                begin_indexes_of_sent_entities = entity.get_entity_begin_idxs()
                for idx in begin_indexes_of_sent_entities:
                    if int(idx) > int(sent.begin_idx) and int(idx) < int(sent.end_idx):
                        sent.add_entity(entity)
                        entity_assigned = True
                        #print(sent.sentence + " was assigned: " + entity.tag + " " + entity.type)
                        break

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
    def get_sentence_obj_list(self):
        return self.sentence_obj_list