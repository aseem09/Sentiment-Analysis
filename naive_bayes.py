

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()

from csv import reader
import math
import pickle
import csv
import re
import math
import random


import pandas as pd
import numpy as np
import string

import spacy
import csv
import re
import pickle

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()



vocabulary = {}
logprior = {}
vocabularyPositive = {}
vocabularyNegative = {}



with open('IMDB_Dataset.csv', 'r') as f:
    file = csv.reader(f)
    imdb_reviews = list(file)

imdb_reviews.pop(0)   
    
random.shuffle(imdb_reviews)

reviews_train = imdb_reviews[0:40000]
reviews = imdb_reviews[40000:]


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags = re.MULTILINE)
    text = re.sub(r'\@\w+|\#', "", text)
    text = re.sub(r"[^a-zA-Z. ]","",text)
    text = re.sub(r'\.+', ".",text)
    text_tokens = word_tokenize(text)
    filtered_words = [word for word in text_tokens if word not in stop_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]  
    # return " ".join(lemma_words)
    return lemma_words
    
positiveCount = 0
negativeCount = 0    

counter = 0

for review in reviews_train:
    counter = counter +1
    print(counter)
    # words = review[0].split(" ")
    words = preprocess_text(review[0])   
    
    #creating vocabulary
    for word in words:
        if word in vocabulary.keys():
            vocabulary[word]= vocabulary[word] + 1
        else:
            vocabulary[word] = 1
    if review[1]=="positive":
        positiveCount = positiveCount + 1
        for word in words:
            if word in vocabularyPositive.keys():
                vocabularyPositive[word] = vocabularyPositive[word] + 1
            else:
                vocabularyPositive[word] = 1    
    else:
        negativeCount = negativeCount + 1
        for word in words:
            if word in vocabularyNegative.keys():
                vocabularyNegative[word] = vocabularyNegative[word] + 1
            else:
                vocabularyNegative[word] = 1 
    
                            
        
#counting positive and negative reviews
                         
totalCount = positiveCount + negativeCount
print(positiveCount)
print(negativeCount)
print(totalCount)

logprior["positive"] = math.log(positiveCount/totalCount)
logprior["negative"] = math.log(negativeCount/totalCount)


#for calculating denominator while calculating likelihood
countWordsInPositive = 0
countWordsInNegative = 0

for word in vocabulary:
    if word in vocabularyPositive.keys():
        countWordsInPositive = countWordsInPositive + vocabularyPositive[word] +1
    else: 
        countWordsInPositive = countWordsInPositive +1
        
    if word in vocabularyNegative.keys():
        countWordsInNegative = countWordsInNegative + vocabularyNegative[word] +1
    else:
        countWordsInNegative = countWordsInNegative +1
        
        
print(logprior)
logLikelihoodPositive = {}   
logLikelihoodNegative = {}

for word in vocabulary.keys():
    if word in vocabularyPositive.keys():
        logLikelihoodPositive[word] = math.log((vocabularyPositive[word]+1)/countWordsInPositive)
    else:
        logLikelihoodPositive[word] = math.log(1/countWordsInPositive)
    
    if word in vocabularyNegative.keys():
        logLikelihoodNegative[word] = math.log((vocabularyNegative[word]+1)/countWordsInNegative)
    else:
        logLikelihoodNegative[word] = math.log(1/countWordsInNegative)
        

with open('vocabulary.txt', 'wb') as file1:
    pickle.dump(vocabulary, file1)
    
with open('logprior.txt', 'wb') as file1:
    pickle.dump(logprior, file1)
    
with open('logLikelihoodPositive.txt', 'wb') as file1:
    pickle.dump(logLikelihoodPositive, file1)
    
with open('logLikelihoodNegative.txt', 'wb') as file1:
    pickle.dump(logLikelihoodNegative, file1)    

print(positiveCount)
print(negativeCount)  

print("training ended")



#testing starts here


tag = "" 
correctPredictions = 0
positivePredictions = 0
negativePredictions = 0
counter = 0

positiveCount = 0
negativeCount = 0
for review in reviews:
    counter = counter +1
    print(counter)
    sumPositive = logprior["positive"]
    sumNegative = logprior["negative"]
#    words = review[0].split(" ")
    words = preprocess_text(review[0])
    for word in words:
        if word in vocabulary:
            sumPositive = sumPositive + logLikelihoodPositive[word]
            sumNegative = sumNegative + logLikelihoodNegative[word]
    if sumPositive>=sumNegative:  
        tag = "positive"
    else:
        tag = "negative"
    if tag == review[1]:
        if tag =="positive":
            positivePredictions = positivePredictions +1
        else:
            negativePredictions = negativePredictions + 1
             
        correctPredictions = correctPredictions + 1
 
accuracyIMDB =  (correctPredictions/10000)*100 
print("sentiment analysis accuracy is ")
print(accuracyIMDB)
print(positivePredictions)
print(negativePredictions)
