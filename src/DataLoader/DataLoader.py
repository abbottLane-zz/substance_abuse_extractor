import os
import fnmatch

class DataLoader:
    def __init__(self, fname):
        self.doc_list = list()
        # Read all Documents into doc_list
        for dirpath, dirs, files in os.walk(fname):
            for filename in fnmatch.filter(files, '*.txt'):
                with open(os.path.join(dirpath, filename)) as f:
                    content = f.readlines()
                    self.doc_list.append(content)

        self.ann_list = list()
        # Read all annotation files into ann_list

        for dirpath, dirs, files in os.walk(fname):
            for filename in fnmatch.filter(files, '*.ann'):
                with open(os.path.join(dirpath, filename)) as f:
                    content = f.readlines()
                    self.doc_list.append(content)


    def get_file_list(self):
        return self.doc_list
    def get_annotations_list(self):
        return  self.ann_list