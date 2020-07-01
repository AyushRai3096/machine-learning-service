from nltk.classify import ClassifierI
from statistics import mode 
import pickle
from nltk.tokenize import word_tokenize
import os
import sys


from django.conf import settings

class VoteClassifier(ClassifierI):

    def __init__(self):

        path = settings.PICKLE_DIR
        classifier_path = os.path.join(path,"classifiers")
        
        self.FileList = get_classifier_pickle_locations_list(classifier_path)
        self.BOW = get_BOW_pickle(path)   

    def get_sentiment(self,sentence):
        features = self.find_features(sentence)
        conf,sentiment = self.classify_and_conf(features)
        return sentiment

    def find_features(self,sentence):
    
        tokens = word_tokenize(sentence)
        features={}
        for word in self.BOW:
            features[word]=(word in tokens)
            
        return features

    def classify_and_conf(self,features):
        votes=[]
        for file in self.FileList:
            with open(file,'rb') as s:
                classifier = pickle.load(s)
                vote = classifier.classify(features)
                votes.append(vote)
        
        choice_vote = votes.count(mode(votes))
        conf = choice_vote / len(votes)
        return conf,mode(votes)

def get_classifier_pickle_locations_list(classifiers_pkl_path):
    res = []
    for filename in os.listdir(classifiers_pkl_path):
        res.append(os.path.join(classifiers_pkl_path,filename))
    return res

def get_BOW_pickle(path):
    BOW = None
    with open(os.path.join(path,"BOW.pickle"),'rb') as s:
        BOW=pickle.load(s)
    return BOW
