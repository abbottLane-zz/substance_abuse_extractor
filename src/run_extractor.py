from Classification import Classifier
from DataLoader import Globals
from DataLoader.DataLoader import DataLoader
from EntityExtractor import EntityExtractor
from FeatureExtractor.FeatureExtractor import FeatureExtractor
from StatusClassification import StatusClassifier
from EventFilling import EventFiller


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

# NOTE: MUST CHANGE PARAMETER stanford_ner_path to your 'stanford-ner.jar' path
STAN_NER_DIR = "C:\\Users\\Spencer\\stanford-ner-2014-06-16\\stanford-ner.jar"

# Train
EntityExtractor.train(training_doc_objs, stanford_ner_path=STAN_NER_DIR)

# Test
EntityExtractor.test(status_classification_info, stanford_ner_path=STAN_NER_DIR)

# DEBUG -- place breakpoint here, take a look at the status_classification object and make sure it has everything we need
test = 0

##################################################################
#### PUTTING EXTRACTION AND STATUS PREDICTIONS TOGETHER #########
################################################################
# DATA FOR COMBINING ATTRIBUTES TO EVENTS:
# Where do I find the CRF classification output?
#   - status_classification.tok_sent_with_crf_classification
#       - This contains a dictionary of {attrib_type:crf output}
#           - attrib_type is in the domain {Temporal, Method, Type, Amount, History}
#           - crf_output is a list of tuples: item1 is the token word, item 2 is the classification prediction
#           -Example: {'History':[(stopped, 0), (smoking, 0), (four, History), (years, History), (ago, History)]}
#
# Where do I find Status Classification output?
#   - status_classification.predicted_event_objs_by_index
#       - This contains a dictionary of {index:list(PredictedEvent objects)}\
#           - index is the index of the sentObj that contains 1 or more PredictedEvents
#           - PredictedEvent is an object carrying all the info about the events we predicted for, and their status

print(status_classification_info)

# Train
attrib_classifier, feature_map = EventFiller.train_event_filler(training_doc_objs)

# Allocate attributes to events
EventFiller.fill_events(status_classification_info, attrib_classifier, feature_map)

print(status_classification_info)
