import os
import fnmatch

from DataLoader.AnnotationDoc import AnnotationDoc
from DataLoader.Document import Document

class DataLoader:
    def __init__(self, fname):
        self.doc_dict = dict()
        # Read all Documents into doc_list
        for dirpath, dirs, files in os.walk(fname):
            for filename in fnmatch.filter(files, "*.txt"):
                with open(os.path.join(dirpath, filename)) as f:
                    content = f.readlines()
                    id = filename[:-4]
                    doc = Document(id, content)
                    self.doc_dict[id]= doc

        self.ann_dict = dict()
        # Read all annotation files into ann_list

        for dirpath, dirs, files in os.walk(fname):
            for filename in fnmatch.filter(files, '*.ann'):
                with open(os.path.join(dirpath, filename)) as f:
                    content = f.readlines()
                    id = filename[:-4]
                    annotations = AnnotationDoc(id, content)
                    self.ann_dict[id]= annotations




    def get_file_dictionary(self):
        return self.doc_dict
    def get_annotations_dictionary(self):
        return  self.ann_dict