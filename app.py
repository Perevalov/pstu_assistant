from flask import Flask, render_template, request
from flask_cors import CORS
from main import assistant


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return assistant.get_answer(userText)


if __name__ == "__main__":
    app.run(debug=True)
