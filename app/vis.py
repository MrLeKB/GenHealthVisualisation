from flask import Flask, render_template
import threading


app = Flask(__name__)


@app.route("/")
def index():
     return render_template("FinalVisualisation.html")

@app.route("/twitter_aug")
def test():
     return render_template("twitter_aug.html")

@app.route("/request_html")
def request_html(year_month):
     print("test")
     return None

@app.route("/initialise_scraper")
def initialise_scraper():
     print("test")
     return None

@app.route("/initialise_analysis")
def initialise_analysis():
     print("test")
     return None

     

