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

## TESTING ###
for key in training_documents.keys():
    print(training_documents[key].get_id())
    for sentence in training_documents[key].get_sentence_obj_list():
        print("\t" + sentence.sentence + "\tSUBS_ABUSE_ENTITY:\t" + str(sentence.has_substance_abuse_entity()))



# Data Loader pipeline output
training_doc_objs = training_documents

##########################################
#### SENTENCE TRAINING PIPELINE #########
########################################