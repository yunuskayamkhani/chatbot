import tkinter as tk
from tkinter import scrolledtext
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import json
import pickle

# Load the model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't get that."

# Create the GUI
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.chatbox = scrolledtext.ScrolledText(root, state='disabled')
        self.chatbox.pack(padx=10, pady=10)
        self.entrybox = tk.Entry(root, width=80)
        self.entrybox.pack(padx=10, pady=10)
        self.entrybox.bind("<Return>", self.send)
        
    def send(self, event=None):
        user_input = self.entrybox.get()
        self.entrybox.delete(0, tk.END)
        self.chatbox.config(state='normal')
        self.chatbox.insert(tk.END, "You: " + user_input + '\n')
        self.chatbox.config(state='disabled')
        self.chatbox.yview(tk.END)
        
        if user_input.lower() != "quit":
            ints = predict_class(user_input)
            response = get_response(ints, intents)
            self.chatbox.config(state='normal')
            self.chatbox.insert(tk.END, "Bot: " + response + '\n')
            self.chatbox.config(state='disabled')
            self.chatbox.yview(tk.END)

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
