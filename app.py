from flask import Flask, render_template, request
from main import assistant


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return assistant.get_answer(userText)


if __name__ == "__main__":
    app.run()
