import nibabel
import os
import re
import numpy
from random import randint

class DatasetBrowser:

    def __init__(self, directoryPath):
        self.dspath = directoryPath
        self.emotionMap = {
            "Look_Neutral_Cue": [0,1,0,0],
            "Look_Neutral_Stim": [0,1,0,0],
            "Look_Neutral_Rating": [0,1,0,0],
            "Look_Neutral_Ant": [0,1,0,0],
            "Reapp_Neg_Cue": [0,0,1,0],
            "Reapp_Neg_Stim": [0,0,1,0],
            "Reapp_Neg_Rating": [0,0,1,0],
            "Reapp_Neg_Ant": [0,0,1,0],
            "Look_Neg_Cue": [0,0,0,1],
            "Look_Neg_Stim": [0,0,0,1],
            "Look_Neg_Rating": [0,0,0,1],
            "Look_Neg_Ant": [0,0,0,1]
        }
    
    def getListOfSubjects(self):
        sublist = []
        pattern = re.compile("^sub-\d+$")
        for e in os.listdir(self.dspath):
            if pattern.match(e):
                sublist.append(e)
        return sublist
    
    def getSubjectData(self, subject):
        x_samples = numpy.zeros(shape=(5,15,64,64,24,1))
        y_samples = numpy.zeros(shape=(5,4))
        pattern = re.compile(".*bold.nii.gz$")
        filesList = os.listdir("{0}/{1}/func/".format(self.dspath, subject))
        a = 0
        for filename in filesList:
            if pattern.match(filename):
                fmriFile = nibabel.load("{0}/{1}/func/{2}".format(self.dspath, subject, filename))
                x_sample_to_add = numpy.array(fmriFile.dataobj).transpose((3,0,1,2))
                x_sample_to_add = x_sample_to_add[:15]
                x_sample_to_add = numpy.reshape(x_sample_to_add, (15,64,64,24,1))
                x_samples[a] = x_sample_to_add
                x_samples = numpy.array(x_samples)
                emotions = [[0,0,0,0]]*15
                emotionFilename = filename.replace("bold.nii.gz", "events.tsv")
                f = open("{0}/{1}/func/{2}".format(self.dspath, subject, emotionFilename))
                raw = f.read()
                lines = raw.split('\n')
                lines.pop(0)
                lines.pop()
                for line in lines:
                    data = line.split('\t')
                    onset, duration, emotion = float(data[0]), float(data[1]), data[2]
                    index = int(onset)//2
                    for i in range(index, int(onset+duration)//2+1):
                        if i < 15:
                            emotions[i] = self.emotionMap[emotion]
                i = randint(0,14)
                y_samples[a] = emotions[i]
        a+= 1
        return (numpy.array(x_samples), numpy.array(y_samples))


if __name__ == "__main__":
    path = "/Users/louislenief/Desktop/Projet IRM/ds000108-00002"
    dsbr = DatasetBrowser(path)
    print("Liste des sujets : ",dsbr.getListOfSubjects())