from DataLoader import Globals


class SentInfo:
    def __init__(self, sent_objs, orig_sents, proc_sents, sent_features, classified_sent_lists):
        self.sent_objs = sent_objs                 # sentence objs
        self.original_sents = orig_sents           # original text, gold labels
        self.processed_sents = proc_sents          # preprocessed text
        self.sent_features = sent_features         # dicts of sentence features to be converted to vector classifiers

        # Structure of sent_lists -- {classifier : set of {indices of sentences classified as such}}
        self.gold_classf_sent_lists = classified_sent_lists  # positive sents for each classification - gold labels
        self.predicted_classf_sent_lists = {}                # positive sents for each classf         - our syste# m
        self.predicted_status = {}

        #Final list of sentence objects for which our system made type and status predictions
        self.predicted_event_objs_by_index = {}         # List of predicted event objs get set after status classification

    # Returns the dicts of sentence features relevant to a particular classifier
    def gold_sent_feats(self, classifier_type):
        sent_indices = []
        sent_feats = []
        labels = []

        # Grab all sentences labelled substance and their specific labels
        for index in self.gold_classf_sent_lists[Globals.SUBSTANCE]:
            sent_indices.append(index)
            sent_feats.append(self.sent_features[index])
            if index in self.gold_classf_sent_lists[classifier_type]:
                labels.append(Globals.HAS_SUBSTANCE)
            else:
                labels.append(Globals.NO_SUBSTANCE)

        return sent_feats, labels

    # Returns the dicts of sentence features relevant to a particular classifier
    def predicted_sent_feats(self, classifier_type):
        sent_indices = []
        sent_feats = []

        if classifier_type == Globals.SUBSTANCE:
            # Grab all sentences and their labels
            for index in range(len(self.original_sents)):
                sent_indices.append(index)
                sent_feats.append(self.sent_features[index])
        else:
            # Grab all sentences labelled substance and their specific labels
            for index in self.predicted_classf_sent_lists[Globals.SUBSTANCE]:
                sent_indices.append(index)
                sent_feats.append(self.sent_features[index])

        return sent_indices, sent_feats

    def get_substance_labels(self, classifier_type):
        labels = []
        for index in range(len(self.processed_sents)):
            if index in self.gold_classf_sent_lists[classifier_type]:
                labels.append(Globals.HAS_SUBSTANCE)
            else:
                labels.append(Globals.NO_SUBSTANCE)
        return labels

    def get_sentences_w_info(self, classifier_type):
        sents = [self.sent_objs[index] for index in self.predicted_classf_sent_lists[classifier_type]]
        return sents

    def get_gold_sentences_w_info(self, classifier_type):
        sents = [self.sent_objs[index] for index in self.gold_classf_sent_lists[classifier_type]]
        return sents
    def get_gold_indexes_w_info(self, classifier_type):
        idxs = [index for index in self.gold_classf_sent_lists[classifier_type]]
        return idxs

    def get_indexes_w_info(self, classifier_type):
        idxs = [index for index in self.predicted_classf_sent_lists[classifier_type]]
        return idxs

    def get_predicted_event_matching_gold_event(self, gold_event, idx):
        if idx not in self.predicted_event_objs_by_index.keys():
            return None
        for event in self.predicted_event_objs_by_index[idx]:
            if event.type == gold_event.type:
                return event
        return None


    def set_predicted_event_dict(self, sent_dict):
        self.predicted_event_objs_by_index = sent_dict

    def evaluate_classifications(self, results_file, test_fold):
        misclass_sents = {}
        out_file = open(results_file, "w")
        out_file.write("\nClassifier Evaluation " + str(test_fold) + "\n------------------------\n")
        for classf in self.predicted_classf_sent_lists:
            misclass_sents[classf] = {}

            # Accuracy
            total = 0
            right = 0
            for index, sent in enumerate(self.original_sents):
                total += 1
                if index in self.predicted_classf_sent_lists[classf] and index in self.gold_classf_sent_lists[classf]:
                    right += 1
                elif index not in self.predicted_classf_sent_lists[classf] and index not in self.gold_classf_sent_lists[classf]:
                    right += 1
                else:
                    misclass_sents[classf][sent] = index in self.gold_classf_sent_lists[classf]
            if total:
                accuracy = right/total
            else:
                accuracy = 0

            # Precision
            n = len([c for c in self.predicted_classf_sent_lists[classf] if (c in self.gold_classf_sent_lists[classf])])
            d = len(self.predicted_classf_sent_lists[classf])
            if d:
                precision = n/d
            else:
                precision = 0

            # Recall
            n = len([c for c in self.gold_classf_sent_lists[classf] if (c in self.predicted_classf_sent_lists[classf])])
            d = len(self.gold_classf_sent_lists[classf])
            if d:
                recall = n/d
            else:
                recall = 0

            # Output Results
            out_file.write("\n<<< " + classf + " >>>\n\tAccuracy:\t" + str(accuracy) + "\n\tPrecision:\t" +
                           str(precision) + "\n\tRecall:  \t" + str(recall) + "\nMislabeled Sentences:\n")

            for sent in misclass_sents[classf]:
                out_file.write(sent + "\n - Should be " + str(misclass_sents[classf][sent]) + "\n")
