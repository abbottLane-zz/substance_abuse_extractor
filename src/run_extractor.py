import pprint
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from DataLoader.DataLoader import DataLoader
from DataLoader.AnnotationDoc import AnnotationDoc
from DataLoader.Document import  Document


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

feature_extractor = FeatureExtractor(training_doc_objs)