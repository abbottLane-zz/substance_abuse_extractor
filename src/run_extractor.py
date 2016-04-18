from DataLoader.DataLoader import DataLoader


data = DataLoader("/home/wlane/PycharmProjects/substance_abuse_extractor/Data")
documents = data.get_file_list()

tmp = 9