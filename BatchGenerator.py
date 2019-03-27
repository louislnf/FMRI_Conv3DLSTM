import keras.utils

from DatasetBrowser import DatasetBrowser
              
class BatchGenerator(keras.utils.Sequence):

    def __init__(self, datasetPath):
        self.dsbr = DatasetBrowser(datasetPath)
        self.subjectsList = self.dsbr.getListOfSubjects()

    def __len__(self):
        return len(self.subjectsList)

    def __getitem__(self, idx):
        return self.dsbr.getSubjectData(self.subjectsList[idx])