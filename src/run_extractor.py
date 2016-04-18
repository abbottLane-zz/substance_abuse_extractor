import pprint

from DataLoader.DataLoader import DataLoader
from DataLoader.AnnotationDoc import AnnotationDoc


data = DataLoader("../Data")
documents = data.get_file_dictionary()
annotations = data.get_annotations_dictionary()

### Test: Make sure all annotation and document id's line up ##
for key in documents.keys():
    document = documents[key]
    annotation = annotations[key]
    print (document.get_id())
    print (annotation.get_id())
### end test ##################################################