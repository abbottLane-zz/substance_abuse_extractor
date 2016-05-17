from Classification import Globals


def train_status_classifiers(sent_info):
    classifiers = dict()

    alcohol_sents = sent_info.get_sentences_w_info(Globals.ALCOHOL)
    drug_sents = sent_info.get_sentences_w_info(Globals.DRUGS)
    tobac_sents = sent_info.get_sentences_w_info(Globals.TOBACCO)

    # Create Feature-Label pairs for each Subs Abuse type
    alc_feats, alc_labels = get_features(alcohol_sents, Globals.ALCOHOL)
    drg_feats, drg_labels = get_features(drug_sents, Globals.DRUGS)
    tbc_feats, tbc_labels = get_features(tobac_sents, Globals.TOBACCO)

    return classifiers

def get_features(sents, type):
    feature_vecs = list()
    labels = list()
    for sent in sents:

        vector = list()
        label,evidence =sent.get_status_label_and_evidence(type)
        input_list = sent.sentence.lower().rstrip(",.!?:;").split()
        some_bigrams = list(get_birgrams(input_list))
        vector.extend([pair[0] + "_" + pair[1] for pair in some_bigrams])
        vector.extend(input_list)
        vector.append("evidence:"+evidence.lower())

        feature_vecs.append(vector)
        labels.append(label)
    return feature_vecs, labels

def get_birgrams(input_list):
    return zip(input_list, input_list[1:])