import pprint
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from DataLoader.DataLoader import DataLoader
from DataLoader.AnnotationDoc import AnnotationDoc
from DataLoader.Document import  Document
from Classification import Classifier
from Classification import Globals

def get_test_fold(folds_data, test_fold_num):

    test_data_docs = set()
    for line in folds_data:
        if "foldName" in line:
            pass
        else:
            fold_name = line.split()[0]
            fold_number = fold_name.split("_")[1]
            file_id = line.split()[1]

            if test_fold_num == int(fold_number):
                test_data_docs.add(file_id[:-4])

    return test_data_docs

####################################
#### DATA LOADER PIPELINE #########
##################################

# load folds data
TEST_FOLD = 4
with open("../Data/folds.out") as file:
   folds_data = file.readlines()
test_set = get_test_fold(folds_data, TEST_FOLD)

# Load data from txt files into memory
data = DataLoader("../Data")

# create dictionaries of {documentId : DocumentObject}
training_documents = data.get_file_dictionary()
annotations = data.get_annotations_dictionary()

# merge annotations data into the documents and split into training and testing sets
training_doc_objs = dict()
testing_doc_objs = dict()

for key in training_documents.keys():
    document = training_documents[key]
    annotation = annotations[key]
    document.set_annotation(annotation)

    if document.get_id() in test_set:
        testing_doc_objs[document.get_id()] = document
    else:
        training_doc_objs[document.get_id()] = document



##########################################
#### SENTENCE TRAINING PIPELINE #########
########################################

# Train classifiers
training_feat_extractor = FeatureExtractor(training_doc_objs)
classifiers, feature_maps = Classifier.train_models(training_feat_extractor)

# Classify sentences
testing_feat_extractor = FeatureExtractor(testing_doc_objs)
sent_classification_info = Classifier.get_classifications(classifiers, feature_maps, testing_feat_extractor)

# How to use:
print("\nSentence Objects with substance info:\n" + str(sent_classification_info.get_sentences_w_info(Globals.SUBSTANCE)))
print("Sentence Objects with alcohol info:\n" + str(sent_classification_info.get_sentences_w_info(Globals.ALCOHOL)))

results_file = "classifier_results.txt"
sent_classification_info.evaluate_classifications(results_file, TEST_FOLD)

##################################
#### EXTRACTION PIPELINE #########
##################################

# Train