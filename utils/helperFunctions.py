import re
import string

from flask import *
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


stopwords = nltk.corpus.stopwords.words('english')

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def processData(data):

    data = remove_punctuation(data)
    data.lower()
    data = tokenization(data)
    data = remove_stopwords(data)
    data = lemmatizer(data)

    return data


def vetorize(data):
    with open('D:\\SentimentAnalysis\\Twitter-Sentiment-Analysis\\tfPickle', 'rb') as f:
        tf = pickle.load(f)
    vectorisedData = tf.transform(data).toarray()
    return vectorisedData


def prediction(data):
    print(data)
    with open('D:\\SentimentAnalysis\\Twitter-Sentiment-Analysis\\modelPickle', 'rb') as f:
        pickleModel = pickle.load(f)
    predictions = pickleModel.predict(data)
    print(predictions[0])
    if predictions[0]==0:
        return 'not-hate'
    else:
        return 'hate'
