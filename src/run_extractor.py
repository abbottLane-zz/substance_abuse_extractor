import pprint
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from DataLoader.DataLoader import DataLoader
from DataLoader.AnnotationDoc import AnnotationDoc
from DataLoader.Document import  Document
from Classification import Classifier


####################################
#### DATA LOADER PIPELINE #########
##################################
# Load data from txt files into memory
training_data = DataLoader("../Data")

# create dictionaries of {documentId : DocumentObject}
training_documents = training_data.get_file_dictionary()
annotations = training_data.get_annotations_dictionary()

# merge annotations data into the documents
for key in training_documents.keys():
    document = training_documents[key]
    annotation = annotations[key]
    document.set_annotation(annotation)

# Data Loader pipeline output
training_doc_objs = training_documents

##########################################
#### SENTENCE TRAINING PIPELINE #########
########################################

# Train classifiers
training_feat_extractor = FeatureExtractor(training_doc_objs)
classifiers, feature_maps = Classifier.train_models(training_feat_extractor)

# Classify sentences
# TODO -- make test doc objs
testing_feat_extractor = FeatureExtractor(training_doc_objs)
sent_classification_info = Classifier.get_classifications(classifiers, feature_maps, testing_feat_extractor)
print(sent_classification_info.gold_classf_sent_lists)
print(sent_classification_info.predicted_classf_sent_lists)