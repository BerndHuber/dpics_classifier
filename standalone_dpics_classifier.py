#! /usr/bin/env python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import re
import cPickle
import datetime
import nltk

def get_sentiment(text):
    text = text.replace("'","")
    text = text.replace("`","")
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def post_process(string):
    string = string.replace("let 's","let us")
    string = string.replace("do n't","don't")
    string = string.replace("i'm","i am")
    string = string.replace("ca n't","can't")
    string = string.replace("cannot","can't")
    string = string.replace("ernie 's","ernie is")
    string = string.replace("we 're","we are")
    string = string.replace("i 'll","i will")
    string = string.replace("we 'll","we will")
    string = string.replace("that 's","that is")
    string = string.replace("you 're","you are")
    string = string.replace("they 're","they are")
    string = string.replace("did n't","didn't")
    string = string.replace("it 's","it is")
    string = string.replace("are n't","are not")
    string = string.replace("was n't","was not")
    string = string.replace("i 'd","i would")
    string = string.replace(" ,","")
    string = string.replace("i 've","i have")
    string = string.replace("how 's","how is")
    string = string.replace("does n't","doesn't")
    string = string.replace("he 's","he's")
    string = string.replace("! "," ")
    string = string.replace(" !"," ")
    string = string.replace("is n't it","isn't it")
    string = string.replace("are n't you","aren't you")
    return string

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return post_process(string.strip().lower())

def get_postags(string):
    return " ".join([str(x[1]) for x in nltk.pos_tag(string.split())])

def classify(utterance):
    #cleaning text
    text = clean_str(str(utterance))

    #get POS tags
    pos = get_postags(text)

    #transform to vectors
    X_dev_counts = count_vect.transform([text]).toarray()
    X_dev_pos = pos_vect.transform([pos]).toarray()

    #get sentiment
    sentiment = get_sentiment(text)

    #concatenate all preprocessed data
    X = np.append(X_dev_counts,X_dev_pos,axis=1)
    X = np.append(X,[[sentiment]],axis=1)

    #classify
    predicted = text_clf_free.predict(X)[0]
    return target_names[predicted-1]

print "loading models.."
analyzer = SentimentIntensityAnalyzer()
with open('tfidf_svm.pkl', 'rb') as fid:
    text_clf_free = cPickle.load(fid)

with open('tfidf_vectorizer_data.pkl', 'rb') as fid:
    count_vect = cPickle.load(fid)

with open('tfidf_vectorizer_pos.pkl', 'rb') as fid:
    pos_vect = cPickle.load(fid)

target_names=['cmd', 'neutral', 'lp', 'qu', 'bd', 'rf', 'nt']

print("starting DPICS classification...")
print "Predicted class:",classify("stop doing that")
print("done!")
