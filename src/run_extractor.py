import pprint
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from DataLoader.DataLoader import DataLoader
from DataLoader.AnnotationDoc import AnnotationDoc
from DataLoader.Document import  Document
from ClassifierTraining import Classifier

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
with open("../Data/folds.out") as file:
   folds_data = file.readlines()
test_set = get_test_fold(folds_data, 1)

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

# Train classifier
training_feat_extractor = FeatureExtractor(training_doc_objs)
classifier, feature_map = Classifier.train_model(training_feat_extractor)

# Filter out sentences w no substance info
testing_feat_extractor = FeatureExtractor(training_doc_objs) # TODO -- make test doc objs
orig_sents_w_subst, proc_sents_w_subst = Classifier.classify_sentences(classifier, feature_map, testing_feat_extractor)
print("Original sentences w substance info:\n\t" + str(orig_sents_w_subst))
print("Processed sentences w substance info:\n\t" + str(proc_sents_w_subst))