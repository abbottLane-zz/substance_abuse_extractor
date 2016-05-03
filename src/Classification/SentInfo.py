from Classification import Globals


class SentInfo:
    def __init__(self, sent_objs, proc_sents, classified_sent_lists):
        self.sent_objs = sent_objs                          # original text, gold labels
        self.processed_sents = proc_sents                   # preprocessed text

        # Structure of sent_lists -- {classifier : set of {indices of sentences classified as such}}
        self.gold_classf_sent_lists = classified_sent_lists  # positive sents for each classification - gold labels
        self.predicted_classf_sent_lists = {}                # positive sents for each classf         - our system

    def get_sentences_w_classf_type(self, classifier_type):
        sent_indices = []
        proc_sents = []
        labels = []

        if classifier_type == Globals.SUBSTANCE:
            # Grab all sentences and their labels
            for index in range(len(self.sent_objs)):
                if index in self.gold_classf_sent_lists[classifier_type]:
                    sent_indices.append(index)
                    proc_sents.append(self.processed_sents[index])
                    labels.append(Globals.HAS_SUBSTANCE)
                else:
                    labels.append(Globals.NO_SUBSTANCE)
        else:
            # Grab all sentences labelled substance and their specific labels
            for index in self.gold_classf_sent_lists[Globals.SUBSTANCE]:
                if index in self.gold_classf_sent_lists[classifier_type]:
                    sent_indices.append(index)
                    proc_sents.append(self.processed_sents[index])
                    labels.append(Globals.HAS_SUBSTANCE)
                else:
                    labels.append(Globals.NO_SUBSTANCE)

        return sent_indices, proc_sents, labels

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
