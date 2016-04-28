class FeatureExtractor:
    def __init__(self, document_objects):
        self.documents= document_objects

        # Feature Extraction Pipeline:
        for key in self.documents:
            print(self.documents[key].get_id())
            for sentence in self.documents[key].get_sentence_obj_list():
                print("\t" + sentence.sentence + "\tSUBSTANCE_ENTITY:\t" + str(sentence.has_substance_abuse_entity()))
