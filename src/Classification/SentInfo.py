from Classification import Globals


class SentInfo:
    def __init__(self, orig_sents, proc_sents, sent_features, classified_sent_lists):
        self.original_sents = orig_sents                 # original text, gold labels
        self.processed_sents = proc_sents          # preprocessed text
        self.sent_features = sent_features         # dicts of sentence features to be converted to vector classifiers

        # Structure of sent_lists -- {classifier : set of {indices of sentences classified as such}}
        self.gold_classf_sent_lists = classified_sent_lists  # positive sents for each classification - gold labels
        self.predicted_classf_sent_lists = {}                # positive sents for each classf         - our system

    # Returns the dicts of sentence features relevant to a particular classifier
    def sent_feats_w_classf_type(self, classifier_type):
        sent_indices = []
        sent_feats = []
        labels = []

        if classifier_type == Globals.SUBSTANCE:
            # Grab all sentences and their labels
            for index in range(len(self.original_sents)):
                if index in self.gold_classf_sent_lists[classifier_type]:
                    sent_indices.append(index)
                    sent_feats.append(self.sent_features[index])
                    labels.append(Globals.HAS_SUBSTANCE)
                else:
                    labels.append(Globals.NO_SUBSTANCE)
        else:
            # Grab all sentences labelled substance and their specific labels
            for index in self.gold_classf_sent_lists[Globals.SUBSTANCE]:
                sent_indices.append(index)
                sent_feats.append(self.sent_features[index])
                if index in self.gold_classf_sent_lists[classifier_type]:
                    labels.append(Globals.HAS_SUBSTANCE)
                else:
                    labels.append(Globals.NO_SUBSTANCE)

        return sent_indices, sent_feats, labels

    '''
    def get_labels(self, classifier_type):
        labels = []

        if classifier_type == Globals.SUBSTANCE:
            # Grab all sentences and their labels
            for index in range(len(self.processed_sents)):
                if index in self.gold_classf_sent_lists[classifier_type]:
                    labels.append(Globals.HAS_SUBSTANCE)
                else:
                    labels.append(Globals.NO_SUBSTANCE)
        else:
            # Grab all sentences labelled substance and their specific labels
            for index in self.gold_classf_sent_lists[Globals.SUBSTANCE]:
                if index in self.gold_classf_sent_lists[classifier_type]:
                    labels.append(Globals.HAS_SUBSTANCE)
                else:
                    labels.append(Globals.NO_SUBSTANCE)

        return labels
    '''

    def get_substance_labels(self, classifier_type):
        labels = []
        for index in range(len(self.processed_sents)):
            if index in self.gold_classf_sent_lists[classifier_type]:
                labels.append(Globals.HAS_SUBSTANCE)
            else:
                labels.append(Globals.NO_SUBSTANCE)
        return labels

    def get_sentences_w_info(self, classifier_type):
        sents = [self.original_sents[index] for index in self.predicted_classf_sent_lists[classifier_type]]
        return sents

    def evaluate_classifications(self):
        # TODO -- do we care about this? Yes we do.
        pass
