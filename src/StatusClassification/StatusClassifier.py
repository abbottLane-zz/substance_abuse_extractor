import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import copy
from DataLoader import Globals
from Classification import SentInfo
from Classification.Classifier import __sentences_and_labels, vectorize_data, vectorize_test_sent, \
    __process_sentence
from DataModels.PredictedEvent import PredictedEvent
from FeatureExtractor.FeatureExtractor import FeatureExtractor


def train_status_classifiers(sent_info):
    classifiers = {}
    feat_maps = {}
    feat_dicts ={}

    alcohol_sents = sent_info.get_gold_sentences_w_info(Globals.ALCOHOL)
    drug_sents = sent_info.get_gold_sentences_w_info(Globals.DRUGS)
    tobac_sents = sent_info.get_gold_sentences_w_info(Globals.TOBACCO)

    #Label each sentence with its type
    for sent in alcohol_sents:
        sent.set_labeled_type(Globals.ALCOHOL)
    for sent in drug_sents:
        sent.set_labeled_type(Globals.DRUGS)
    for sent in tobac_sents:
        sent.set_labeled_type(Globals.TOBACCO)

    # Create Feature-Label pairs for each Subs Abuse type
    alc_feats, alc_labels = get_features(alcohol_sents)
    drg_feats, drg_labels = get_features(drug_sents)
    tbc_feats, tbc_labels = get_features(tobac_sents)

    feat_dicts["Alcohol"] =alc_feats
    feat_dicts["Drug"] = drg_feats
    feat_dicts["Tobacco"]=tbc_feats

    # Create Model
    alc_classifier, alc_feature_map = train_model(alc_feats, alc_labels)
    drg_classifier, drg_feature_map = train_model(drg_feats, drg_labels)
    tbc_classifier, tbc_feature_map = train_model(tbc_feats, tbc_labels)

    # Set classifier in dictionary
    classifiers[Globals.ALCOHOL] = alc_classifier
    classifiers[Globals.DRUGS] = drg_classifier
    classifiers[Globals.TOBACCO] = tbc_classifier

    # Set feature map in dictionary
    feat_maps[Globals.ALCOHOL] = alc_feature_map
    feat_maps[Globals.DRUGS] = drg_feature_map
    feat_maps[Globals.TOBACCO] = tbc_feature_map

    return classifiers, feat_maps, feat_dicts

def get_features(sents):
    feature_vecs = list()
    labels = list()
    for sent in sents:

        vector = dict()
        label,evidence =sent.get_status_label_and_evidence(sent.labeled_type)
        input_list = sent.sentence.lower().rstrip(",.!?:;").split()
        some_bigrams = list(get_bigrams(input_list))

        for pair in some_bigrams:
            vector[pair[0] + "_" + pair[1]] = True
        for x in input_list:
            vector[x]=True

        # This don't work, test data needs access to any features you choose
        #vector["evidence:"+evidence.lower()]=True

        feature_vecs.append(vector)
        labels.append(label)
    return feature_vecs, labels

def get_bigrams(input_list):
    return zip(input_list, input_list[1:])

def train_model(proc_sents, labels):
        # Convert Data to vectors
        sent_vectors, labels_for_classifier, feature_map = vectorize_data(proc_sents, labels)

        # Create Model
        classifier = LinearSVC()
        classifier.fit(sent_vectors, labels_for_classifier)

        return classifier, feature_map


def get_classifications(status_classifiers, status_feat_maps, feats_dicts, sent_info):


    # Specific classifications
    for classf in status_classifiers:
        sent_info = classify(status_classifiers[classf], classf, status_feat_maps[classf], sent_info)

    return sent_info

def classify(classifier, classifier_type, feature_map, sent_info):

    relevent_sentences = sent_info.get_sentences_w_info(classifier_type)

    features, labels = get_features(relevent_sentences)
    number_of_sentences = len(features)
    number_of_features = len(feature_map)

    # Vectorize sentences and classify
    test_vectors = [vectorize_test_sent(feats, feature_map) for feats in features]
    test_array = np.reshape(test_vectors, (number_of_sentences, number_of_features))
    classifications = classifier.predict(test_array)

    sent_info.predicted_status[classifier_type] = classifications

    return sent_info


def print_exp_precision_recall_by_type(status_info, out_file, type):
    # Write sentence classification info to out_file for debugging
    out_file.write("+++++++++++++++++\n")
    out_file.write("++++  TYPE: " + type + "++++\n")
    out_file.write("+++++++++++++++++\n")
    for event_list in status_info.predicted_event_objs_by_index.values():
        for event in event_list:
            if event.type == type:
                gold_sentence = status_info.sent_objs[event.sent_obj_idx]
                sentence = gold_sentence.sentence
                for gold_event in gold_sentence.set_entities:
                    if gold_event.type == type:
                        actual = gold_event.get_status()
                        predicted = event.status
                        out_file.write(sentence + "\n")
                        out_file.write("\tTYPE PREDICTION: " + event.type + "\n")
                        out_file.write("\tTYPE ACTUAL: " + gold_event.type + "\n")
                        out_file.write("\tSTATUS PREDICTION: " + predicted + "\n")
                        out_file.write("\tSTATUS ACTUAL: " + actual+ "\n")

    #RECALL - # correct 'type,status' pairs / all possible entities in gold of 'type'
    total_type_events = 0
    correct =0
    for idx, sent in enumerate(status_info.sent_objs):
        for gold_event in sent.set_entities:
            if gold_event.type == type: # if Gold event is of the type we are currently calculating for
                total_type_events += 1
                predicted_event = status_info.get_predicted_event_matching_gold_event(gold_event, idx)    # fetch predicted event for current gold event
                if predicted_event != None and predicted_event.type == gold_event.type and predicted_event.status == gold_event.get_status():
                    correct+=1

    #precision = tp / float(tp + fp)
    recall = correct / float(total_type_events)


    # Precision
    total_events_predicted=0
    correct = 0
    for idx, event_list in enumerate(status_info.predicted_event_objs_by_index.values()):
        for event in event_list:
            if event.type == type:
                gold_sentence = status_info.sent_objs[event.sent_obj_idx]
                for gold_event in gold_sentence.set_entities:
                    if gold_event.type == type:
                        total_events_predicted += 1
                        if event.type == gold_event.type and event.status == gold_event.get_status():
                            correct +=1

    precision=correct/total_events_predicted
    print("==========" + type + " Status Classification ===========")
    print("PRECISION:")
    print(precision)
    print("RECALL:")
    print(recall)
    print("=================================================\n")
    pass


def evaluate_status_classification(status_info, status_result_file, TEST_FOLD):
    out_file = open(status_result_file, "w")
    out_file.write("\nStatus Classifier Evaluation, tested on fold " + str(TEST_FOLD) + "\n------------------------\n")

    # FP/TP/FN Counts
    for type in Globals.SPECIFIC_CLASSIFIER_TYPES:
        print_exp_precision_recall_by_type(status_info, out_file, type)


def finalize_classification_info_object(status_info):
    predicted_event_dict = {}
    for type in status_info.predicted_status:
        predicted_indexes = status_info.get_indexes_w_info(type)
        predicted_status = status_info.predicted_status[type]

        for idx, status in zip(predicted_indexes, predicted_status):
            if idx not in predicted_event_dict.keys():
                predicted_event_dict[idx] = list()
            if idx in predicted_indexes:  # If the global list found something we tried to predict for
                pred_status = status
                sentence = status_info.sent_objs[idx]
                pred_event = PredictedEvent(type, pred_status, sentence, idx)
                predicted_event_dict[idx].append(pred_event)

        # set the newly-created dict back into the sent_info obj
        status_info.set_predicted_event_dict(predicted_event_dict)
    return status_info




