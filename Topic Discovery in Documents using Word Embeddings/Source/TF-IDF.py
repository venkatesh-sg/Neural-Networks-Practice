import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy
import string
from nltk.stem.porter import PorterStemmer


Directory="Data/" # Directory for the text files

# Query to be analysed on the documents
Query="cancer"

# Tokenizer and Stemmer for words
stemmer = PorterStemmer()
def tokenizer(text):
    words = nltk.word_tokenize(text)
    tokens = [i for i in words if i not in string.punctuation]
    stems = [stemmer.stem(i) for i in tokens]
    return stems


for subdir, dirs, files in os.walk(Directory):

    # Caluculate TF-IDF over all the documents
    print("Initializing vectorizer")
    vectorizer = TfidfVectorizer(tokenizer=tokenizer,stop_words='english', binary=False,dtype=numpy.float)
    Tfidf= vectorizer.fit_transform(open(os.path.join(subdir, file)).read() for file in files)
    print("Calculated TFIDF matrix")

    # Tokenize and stem Query
    QueryTokens=tokenizer(Query)
    QTokenIndecs=[vectorizer.vocabulary_[i] for i in QueryTokens]
    print("Query Tokenizer")

    # Convert the matrix into array
    TFMat=Tfidf.toarray()
    print("Convert matrix to array")


    # Adding the tf-idf score for all the tokens of the query
    Final=numpy.zeros(shape=(len(files)))
    print("Calculating Final weights")
    for i in QTokenIndecs:
        Final=Final+TFMat[:,i]
    # Get the top documents in the corpus
    Top10_documents=Final.argsort()[-10:][::-1]

    print("Top 10 files that would give info about Query are")
    for i in Top10_documents:
        print(files[i])