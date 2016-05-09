from FeatureExtractor.FeatureExtractor import FeatureExtractor
from Classification import Globals
import subprocess

features = [
    "useClassFeature=true",
    "useWord=true",
    "useNGrams=true",
    "noMidNGrams=true",
    "useDisjunctive=true",
    "maxNGramLeng=6",
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


def classify(stanford_ner_path, train_file_name, prop_file_name, shell_script_name):
    '''
    This classifies.
    :return: list of sent_objs with attributes filled
    '''
    global features
    # create_prop_file(prop_file_name, features)
    train_model(stanford_ner_path, "austen.prop", shell_script_name)


def train_model(stanford_ner_path, prop_file_name, shell_script_name):
    subprocess.call(["./"+shell_script_name+" "+stanford_ner_path+" "+prop_file_name], shell=True)


def create_train_file(train_sents, train_file_name):
    train_file = open(train_file_name, 'w')
    for sent in train_sents:
        pass
    train_file.close()


def create_prop_file(prop_file_name, features):
    prop_file = open(prop_file_name, 'w')
    prop_file.write("trainFile = " + train_file_name + "\n")
    prop_file.write("serializeTo = ner-model.ser.gz\n")
    # prop_file.write("map = word=0,status=1,temp=2,method=3,type=4,amount=5,freq=6,hist=7\n")
    prop_file.write("map = word=0,status=1\n")
    for feat in features:
        prop_file.write(feat + "\n")
    prop_file.close()


if __name__ == "__main__":
    stanford_ner_path = "/Users/Martin/stanford-ner-2015-04-20/stanford-ner.jar"
    train_file_name = "jane-austen-emma-ch1.tsv"
    prop_file_name = "attr_extract.prop"
    shell_script_name = "train_model.sh"
    classify(stanford_ner_path, train_file_name, prop_file_name, shell_script_name)
