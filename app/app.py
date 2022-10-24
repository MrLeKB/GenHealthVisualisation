from flask import Flask, render_template
import threading


app = Flask(__name__)


@app.route("/")
def index():
     t = threading.Thread(target=home)
     t2 = threading.Thread(target=func)
     t.start()
     t2.start()
     # t.join()
     # t2.join()



def func():
     print("thread created____________________")
     print("Task 1 assigned to thread: {}".format(threading.current_thread().name))     

@app.route("/home")
def home():
     print("Vis assigned to thread: {}".format(threading.current_thread().name))  
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

     
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
