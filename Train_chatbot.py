import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# Initialize NLTK data path and download necessary resources
nltk.data.path.append('C:\\Users\\ykkhani\\AppData\\Roaming\\nltk_data')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# Step 2: Preprocessing the Data

# Initialize empty lists
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Loop through each intent in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        
        # Add the pattern and the associated tag to documentsh
        documents.append((word_list, intent['tag']))
        
        # Add the tag to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word, and remove duplicates
# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))  # Use sorted to sort the list of unique words

