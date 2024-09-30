# app.py
from dotenv import load_dotenv
from chatbot_app import ChatbotApp

if __name__ == "__main__":
    load_dotenv()
    app = ChatbotApp()
    app.run()
