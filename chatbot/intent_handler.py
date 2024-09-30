# intent_handler.py
import json
import random

class IntentHandler:
    def __init__(self, intents_path):
        with open(intents_path, encoding='utf-8') as file:
            self.intents = json.load(file)

    def get_response(self, ints):
        tag = ints[0]["intent"]
        list_of_intents = self.intents["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                return random.choice(i["responses"])
        return "Lo siento, no entendí eso."
