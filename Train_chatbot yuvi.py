import numpy as np
2
from keras.models import Sequential
3
from keras.layers import Dense, Activation, Dropout
4
from keras.optimizers import SGD
5
import random
6
7
import nltk
8
from nltk.stem import WordNetLemmatizer
9
lemmatizer = WordNetLemmatizer()
10
import json
11
import pickle
12
13
intents_file = open('intents.json').read()
14
intents = json.loads(intents_file)

# Initialize lists to hold words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Iterate over each intent in the intents JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the documents to the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes if the intent tag is not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# The result: words is a list of unique lemmatized words, and classes is a list of unique intent tags.
