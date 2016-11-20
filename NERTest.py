# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:18:27 2016

@author: adam
"""

import nltk
from nltk.tag import StanfordNERTagger
import pickle

def loadHeadlines():
    return pickle.load(open('articleHeadlines.p','rb'))


#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
stanford_dir = '/home/adam/stanford-ner-2015-12-09/'
jarfile = stanford_dir + 'stanford-ner.jar'
modelfile = stanford_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'

st = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)

res = st.tag('Rami Eid is studying at Stony Brook University in NY'.split())


tests = loadHeadlines()
partialTest = tests[-50::]
NER_Results = []
counter = 0
for test in partialTest:
    temp = st.tag(test.split())
    print(counter)
    NER_Results.append(temp)
    counter += 1