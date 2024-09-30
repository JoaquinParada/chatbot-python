# model.py
import numpy as np
import pickle
from keras.models import load_model

class ChatbotModel:
    def __init__(self, model_path, words_path, classes_path):
        self.model = load_model(model_path)
        self.words = pickle.load(open(words_path, "rb"))
        self.classes = pickle.load(open(classes_path, "rb"))

    def clean_up_sentence(self, sentence):
        from nltk.stem import WordNetLemmatizer
        import nltk
        lemmatizer = WordNetLemmatizer()
        
        sentence_words = nltk.word_tokenize(sentence)
        return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]
        return return_list
