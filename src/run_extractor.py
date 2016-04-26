import pprint
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from DataLoader.DataLoader import DataLoader
from DataLoader.AnnotationDoc import AnnotationDoc
from DataLoader.Document import  Document


####################################
#### DATA LOADER PIPELINE #########
##################################
# Load data from txt files into memory
data = DataLoader("../Data")

# create dictionaries of {documentId : DocumentObject}
documents = data.get_file_dictionary() #TODO: make sure sentences are being properly build into objects
annotations = data.get_annotations_dictionary()

# merge annotations data into the documents
for key in documents.keys():
    document = documents[key]
    annotation = annotations[key]
    document.set_annotation(annotation)

# Data Loader pipeline output
training_doc_objs = documents

##########################################
#### SENTENCE TRAINING PIPELINE #########
########################################