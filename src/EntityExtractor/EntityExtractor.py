from FeatureExtractor.FeatureExtractor import FeatureExtractor
from Classification import Globals
import subprocess
# from nltk.tag import StanfordNERTagger
import re


features = [
    "useClassFeature=true",
    "useWord=true",
    "useNGrams=true",
    "noMidNGrams=true",
    "useDisjunctive=true",
    "maxNGramLeng=3",
    "usePrev=true",
    "useNext=true",
    "useSequences=true",
    "usePrevSequences=true",
    "maxLeft=1",
    "useTypeSeqs=true",
    "useTypeSeqs2=true",
    "useTypeySequences=true",
    "wordShape=chris2useLC"
]

entity_types = [
    "Status",
    "Temporal",
    "Method",
    "Type",
    "Amount",
    "Frequency",
    "History"
]


def classify(training_doc_objs,
             stanford_ner_path="/home/wlane/stanford-ner-2015-04-20/stanford-ner.jar",
             train_file_name="train-file.tsv",
             prop_file_name="attr_extract.prop",
             model_name="ner-model.ser.gz",
             test_file_name="test-file.tsv",
             train_script_name="train_model.sh",
             test_script_name="test_classify.sh"):
    """
    This function is currently not used.
    :return: list of sent_objs with attributes filled
    """
    global features
    create_train_file(training_doc_objs, train_file_name)
    create_prop_file(prop_file_name, train_file_name, features, model_name)
    train_model(stanford_ner_path, prop_file_name, train_script_name)
    test_model(stanford_ner_path, model_name, test_file_name, test_script_name)


def get_model_from_full_id(model_name):
    sections= model_name.split("-")
    last_section = sections[len(sections)-1]
    type_ser_gz = last_section.split(".")
    type = type_ser_gz[0]
    return type


def test(testing_doc_objs, path="EntityExtractor/",
         stanford_ner_path="/home/wlane/stanford-ner-2015-04-20/stanford-ner.jar",
         test_script_name="test_classify.sh"):
    global entity_types
    for type in entity_types:
        if type != "Status":
            curr_doc_num=1
            for docobj in testing_doc_objs.values():
                for sentobj in docobj.get_sentence_obj_list():
                    list_of_one_sentence_because_a_list_is_whats_expected_here = list()
                    list_of_one_sentence_because_a_list_is_whats_expected_here.append(sentobj)
                    test_file_name = path + "Test-Files/" + "test-" +sentobj.id.replace("-", "")+ "_"+ type + ".tsv"
                    model_name = path + "Models/" + "model-" + type + ".ser.gz"
                    create_test_file(list_of_one_sentence_because_a_list_is_whats_expected_here, test_file_name, type)
                    model_to_use_for_classification = get_model_from_full_id(model_name)
                    print("Classifying Sentence ID: " +test_file_name + " with CRF model: " + model_to_use_for_classification)
                    test_model(stanford_ner_path,model_name, test_file_name, path+test_script_name)


def train(training_doc_objs, path="EntityExtractor/",
          stanford_ner_path="/home/wlane/stanford-ner-2015-04-20/stanford-ner.jar",
          train_script_name="train_model.sh"):
    global features
    global entity_types
    for type in entity_types:
        if type != "Status":
            train_file_name = path + "Train-Files/" + "train-" + type + ".tsv"
            prop_file_name = path + "Prop-Files/" + type + ".prop"
            model_name = path + "Models/" + "model-" + type + ".ser.gz"
            create_train_file(training_doc_objs, train_file_name, type)
            create_prop_file(prop_file_name, train_file_name, features, model_name)
            train_model(stanford_ner_path, prop_file_name, path+train_script_name)


def test_model(stanford_ner_path, model_name, test_file_name, test_script_name):
    subprocess.call(["./" + test_script_name + " " + stanford_ner_path + " " + model_name + " " + test_file_name], shell=True)


def train_model(stanford_ner_path, prop_file_name, train_script_name):
    subprocess.call(["./"+train_script_name+" "+stanford_ner_path+" "+prop_file_name], shell=True)

def create_test_file(list_sent_objs, test_file_name, type):
    test_file = open(test_file_name, 'w')
    for sent_obj in list_sent_objs:
        #if sent_obj.has_substance_abuse_entity(): # is this legal? this method checks gold label status
        sentence = sent_obj.sentence
        # Debug lines
        # train_file.write(doc + "\n")
        # train_file.write(sentence + "\n")
        entity_set = sent_obj.set_entities
        sent_offset = sent_obj.begin_idx
        for match in re.finditer("\S+", sentence):
            start = match.start()
            end = match.end()
            pointer = sent_offset + start
            word = match.group(0)
            test_file.write(word.rstrip(",.:;"))
            # Debug line
            # train_file.write("[" + str(pointer) + "," + str(sent_offset + match.end()) + "]")
            test_file.write("\t")
            answer = "0"
            debug_str = ""
            for entity in entity_set:
                if answer != "0":
                    break
                if entity.is_substance_abuse():
                    attr_dict = entity.dict_of_attribs
                    for attr in attr_dict:
                        attr_start = int(attr_dict[attr].span_begin)
                        attr_end = int(attr_dict[attr].span_end)
                        if attr_dict[attr].type == type and \
                           attr_start <= pointer < attr_end:
                            answer = type
                            # Debug lines
                            # answer += "\t" + attr_dict[attr].text +\
                            #          "[" + str(attr_start) +\
                            #          "," + str(attr_end) + "]"
                            debug_str = "--- Sent obj start index: " + str(sent_offset) + "\n" + \
                                        "--- Match obj start index: " + str(start) + "\n" + \
                                        "--- Match obj end index: " + str(end) + "\n" + \
                                        "--- Pointer index: " + str(sent_offset) + " + " + \
                                        str(start) + " = " + str(pointer) + "\n" + \
                                        "--- Attr start index: " + str(attr_start) + "\n" + \
                                        "--- Attr end index: " + str(attr_end) + "\n"
                            break
            test_file.write(answer + "\n")
            # Debug line
            # train_file.write(debug_str)
            #print(debug_str)
    test_file.close()

def create_train_file(training_doc_objs, train_file_name, type):
    """
    Sorry about the crazy embedded FOR loops and indents.
    I will modularize better to make it prettier.
    """
    train_file = open(train_file_name, 'w')
    for doc in training_doc_objs:
        doc_obj = training_doc_objs[doc]
        for sent_obj in doc_obj.get_sentence_obj_list():

            #Debug print:
            # if "2 to 3 packets per day for at least" in sent_obj.sentence:
            #     test = 0

            if sent_obj.has_substance_abuse_entity():
                sentence = sent_obj.sentence
                # Debug lines
                # train_file.write(doc + "\n")
                # train_file.write(sentence + "\n")
                entity_set = sent_obj.set_entities
                sent_offset = sent_obj.begin_idx
                for match in re.finditer("\S+", sentence):
                    start = match.start()
                    end = match.end()
                    pointer = sent_offset + start
                    word = match.group(0)
                    train_file.write(word.rstrip(",.:;"))
                    # Debug line
                    # train_file.write("[" + str(pointer) + "," + str(sent_offset + match.end()) + "]")
                    train_file.write("\t")
                    answer = "0"
                    debug_str = ""
                    for entity in entity_set:
                        if answer != "0":
                            break
                        if entity.is_substance_abuse():
                            attr_dict = entity.dict_of_attribs
                            for attr in attr_dict:
                                attr_start = int(attr_dict[attr].span_begin)
                                attr_end = int(attr_dict[attr].span_end)
                                if attr_dict[attr].type == type and \
                                   attr_start <= pointer < attr_end:
                                    answer = type
                                    # Debug lines
                                    # answer += "\t" + attr_dict[attr].text +\
                                    #          "[" + str(attr_start) +\
                                    #          "," + str(attr_end) + "]"
                                    debug_str = "--- Sent obj start index: " + str(sent_offset) + "\n" + \
                                                "--- Match obj start index: " + str(start) + "\n" + \
                                                "--- Match obj end index: " + str(end) + "\n" + \
                                                "--- Pointer index: " + str(sent_offset) + " + " + \
                                                str(start) + " = " + str(pointer) + "\n" + \
                                                "--- Attr start index: " + str(attr_start) + "\n" + \
                                                "--- Attr end index: " + str(attr_end) + "\n"
                                    break
                    train_file.write(answer + "\n")
                    # Debug line
                    # train_file.write(debug_str)
                    #print(debug_str)

    train_file.close()


def create_prop_file(prop_file_name, train_file_name, features, model_name):
    prop_file = open(prop_file_name, 'w')
    prop_file.write("trainFile = " + train_file_name + "\n")
    prop_file.write("serializeTo = " + model_name + "\n")
    # prop_file.write("map = word=0,answer=1,temp=2,method=3,type=4,amount=5,freq=6,hist=7\n")
    prop_file.write("map = word=0,answer=1\n")
    for feat in features:
        prop_file.write(feat + "\n")
    prop_file.close()


if __name__ == "__main__":
    stanford_ner_path = "/home/wlane/stanford-ner-2015-04-20/stanford-ner.jar"
    # train_file_name = "jane-austen-emma-ch1.tsv"
    # train_file_name = "dummy-train.tsv"
    train_file_name = "dummer.tsv"
    test_file_name = "dummer-test.tsv"
    # test_file_name = "dummy-train.tsv"
    # prop_file_name = "austen.prop"
    prop_file_name = "attr_extract.prop"
    model_name = "ner-model.ser.gz"
    train_script_name = "train_model.sh"
    test_script_name = "test_classify.sh"
    # classify(stanford_ner_path, train_file_name, prop_file_name, model_name,
    #          test_file_name, train_script_name, test_script_name)
