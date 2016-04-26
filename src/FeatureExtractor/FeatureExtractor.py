class FeatureExtractor:
    def __init__(self, documents, annotation_documents):
        self.documents= documents
        self.annotation_documents = annotation_documents
        self.feature_vector = list()

        # Feature Extraction Pipeline:
        #   For each document:
        #       extract the gold standard label
        #       process the raw data here to automatically extract features and put them in the feature_vector list
        for key in documents.keys():



            self.feature_vector.append("")




        pass
