from Classification import Classifier
from DataLoader import Globals
from DataLoader.DataLoader import DataLoader
from EntityExtractor import EntityExtractor
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from StatusClassification import StatusClassifier


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
TEST_FOLD = 1
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

# Print testing data stats
print("==== Testing Data Stats ====")
count=0
num_sents=0
for doc in testing_doc_objs.values():
    for sent in doc.sentence_obj_list:
        num_sents+=1
        for event in sent.set_entities:
            if event.type in Globals.SPECIFIC_CLASSIFIER_TYPES:
                count +=1
print("NUMBER OF SUBSTANCE ABUSE EVENTS:"+ str(count))
print("NUMBER OF SENTENCES: " + str(num_sents))
##########################################
#### EVENT CLASSIFICATION       #########
########################################

# Train classifiers
training_feat_extractor = FeatureExtractor(training_doc_objs)
classifiers, feature_maps, sent_info = Classifier.train_models(training_feat_extractor)

# Classify sentences - Substance abuse general, and abuse type classifications
testing_feat_extractor = FeatureExtractor(testing_doc_objs)
sent_classification_info = Classifier.get_classifications(classifiers, feature_maps, testing_feat_extractor)

# How to use:
print("\nSentence Objects with substance info:\n" + str(sent_classification_info.get_sentences_w_info(
    Globals.SUBSTANCE)))
print("Sentence Objects with alcohol info:\n" + str(sent_classification_info.get_sentences_w_info(Globals.ALCOHOL)))

results_file = Globals.CLASSF_EVAL_FILE
sent_classification_info.evaluate_classifications(results_file, TEST_FOLD)

########################################
####   STATUS CLASSIFICATION ##########
######################################
training_input = sent_info
testing_input = sent_classification_info

# Train status
status_classifiers, status_feat_maps, feats_dicts = StatusClassifier.train_status_classifiers(training_input)

# Classify status
status_classification_info = StatusClassifier.get_classifications(status_classifiers, status_feat_maps, feats_dicts, testing_input)

# Combine predicted type and status into status_classification_info
status_classification_info = StatusClassifier.finalize_classification_info_object(status_classification_info)

status_result_file = "status_results.txt"

StatusClassifier.evaluate_status_classification(status_classification_info, status_result_file, TEST_FOLD)

############################################
#### ATTRIBUTE EXTRACTION PIPELINE #########
############################################

# Train
# NOTE: MUST CHANGE PARAMETER stanford_ner_path to your 'stanford-ner.jar' path
# EntityExtractor.train(training_doc_objs, stanford_ner_path="/home/wlane/stanford-ner-2014-06-16/stanford-ner.jar")

# Test
#   (Currently uses gold standard for choosing substance abuse sentences instead
#   of classification due to local issues with scipy; I will change this soon)
# NOTE: MUST CHANGE PARAMETER stanford_ner_path to your 'stanford-ner.jar' path
# EntityExtractor.test(testing_doc_objs, stanford_ner_path="/home/wlane/stanford-ner-2014-06-16/stanford-ner.jar")


#####################################
#### EVENT FILLING PIPELINE #########
#####################################

