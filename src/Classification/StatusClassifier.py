from Classification import Globals


def train_status_classifiers(sent_info):
    classifiers = dict()

    alcohol_sents = sent_info.get_sentences_w_info(Globals.ALCOHOL)
    drug_sents = sent_info.get_sentences_w_info(Globals.DRUGS)
    tobac_sents = sent_info.get_sentences_w_info(Globals.TOBACCO)

    # Train classifier for each abuse type
    alc_feats, alc_labels = sent_info.gold_sent_feats(Globals.ALCOHOL)
    drg_feats, drg_labels = sent_info.gold_sent_feats(Globals.DRUGS)
    tbc_feats, tbc_labels = sent_info.gold_sent_feats(Globals.TOBACCO)

    return classifiers