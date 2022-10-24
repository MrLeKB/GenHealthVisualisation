from flask import Flask, render_template
import threading


app = Flask(__name__)

t = threading.Thread(target=func)
t.start()

@app.route("/")
def index():
     return render_template("FinalVisualisation.html")

def func():
     print("__________Task 1 assigned to thread: {}".format(threading.current_thread().name))     


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

     
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
