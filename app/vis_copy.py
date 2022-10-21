from flask import Flask, render_template

app2 = Flask(__name__)

@app2.route("/")
def index():
     return render_template("FinalVisualisation.html")

@app2.route("/twitter_aug")
def test():
     return render_template("twitter_aug.html")


