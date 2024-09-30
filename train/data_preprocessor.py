# train/data_preprocessor.py
import json
import nltk
from nltk.stem import WordNetLemmatizer
import os
import pickle

class DataPreprocessor:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ["?", "!"]
        
    def load_intents(self):
        with open(self.intents_path, encoding='utf-8') as file:
            intents = json.load(file)
        return intents
    
    def process_data(self):
        intents = self.load_intents()

        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent["tag"]))
                
                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])
        
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        self.save_data()
        
    def save_data(self):
        pickle.dump(self.words, open("words.pkl", "wb"))
        pickle.dump(self.classes, open("classes.pkl", "wb"))
