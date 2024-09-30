# train/trainer.py
import os
from train.data_preprocessor import DataPreprocessor
from train.model_trainer import ModelTrainer

class Trainer:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.data_preprocessor = DataPreprocessor(intents_path)
        self.data_preprocessor.process_data()
        
        self.words = self.data_preprocessor.words
        self.classes = self.data_preprocessor.classes
        self.documents = self.data_preprocessor.documents
        
        self.model_trainer = ModelTrainer(self.words, self.classes, self.documents)

    def train_model(self):
        train_x, train_y = self.model_trainer.prepare_training_data()
        model = self.model_trainer.build_model(train_x, train_y)
        hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
        model.save("chatbot_model.h5")
        print("Model created")
