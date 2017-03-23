"""
1 - Select favourite book
2 - Check manually if Goodreads has reviews on this book
3 - Either study the API to pull information or scrape reviews 
4 - Use standard term frequency approaches(TFIDF) to extract meaningful terminology from the reviews.
5 - Review you approach. What challenges do you see?
6 - Is there a better way to do this task?


Find most meaningful words
"""

import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

token_dict = []
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if(item!=','and item!='+' and item!='.' and item!=')' and item!='(' and item!="''" and item!="'s"):
            stemmed.append(stemmer.stem(item))

    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def tfidf_max(values):
    temp = []
    for i in range(len(values)):
        temp.append(values[i][1])
    maxvalue = max(temp)

    for i in range(len(values)):
        if(values[i][1]==maxvalue):
            print(values[i])



with open("reviews.txt", 'r') as f:
    content = f.readlines()

for file in content:
    lowers = file.lower()
    no_punctuation = lowers.translate(string.punctuation)
    token_dict.append(no_punctuation)
        
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict)



feature_names = tfidf.get_feature_names()
tfs_array=tfs.toarray()[0].tolist()


values = []
for i in range(0, len(feature_names)):
    couple = []
    if(tfs_array[i]!= 0.0):
        couple.append(feature_names[i])
        couple.append(tfs_array[i])
        values.append(couple)

"""
for i in values:
    print(i)
"""

tfidf_max(values)
