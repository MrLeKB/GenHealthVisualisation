from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():

     print("first thread______________________________________________")
     return render_template("FinalVisualisation.html")

@app.route("/twitter_aug")
def test():
     return render_template("twitter_aug.html")


