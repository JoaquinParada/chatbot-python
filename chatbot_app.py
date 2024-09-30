# chatbot.py
from flask import Flask, render_template, request
from chatbot.model import ChatbotModel
from chatbot.intent_handler import IntentHandler
import os

class ChatbotApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.model = ChatbotModel("chatbot_model.h5", "words.pkl", "classes.pkl")
        self.intent_handler = IntentHandler(os.getenv('INTENTS_PATH'))

        @self.app.route("/")
        def home():
            return render_template("./index.html")

        @self.app.route("/get", methods=["POST"])
        def chatbot_response():
            msg = request.form["msg"]
            ints = self.model.predict_class(msg)
            res = self.intent_handler.get_response(ints)
            return res

    def run(self):
        self.app.run()
