# train/model_trainer.py
import numpy as np
import random
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD

class ModelTrainer:
    def __init__(self, words, classes, documents):
        self.words = words
        self.classes = classes
        self.documents = documents
        
    def prepare_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [word.lower() for word in pattern_words]
            
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        random.shuffle(training)
        train_x = np.array([item[0] for item in training])
        train_y = np.array([item[1] for item in training])
        
        return train_x, train_y

    def build_model(self, train_x, train_y):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation="softmax"))
        
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
        
        return model
