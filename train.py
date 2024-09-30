# train.py
from train.trainer import Trainer
import os
from dotenv import load_dotenv

load_dotenv()
intents_path = os.getenv('INTENTS_PATH')

if __name__ == "__main__":
    trainer = Trainer(intents_path)
    trainer.train_model()
