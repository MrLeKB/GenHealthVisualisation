from flask import Flask, render_template
import threading
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import Error
import pandas as pd


app = Flask(__name__)

engine = create_engine('postgresql+psycopg2://pvahbfuxwqhvpq:3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb@ec2-34-194-40-194.compute-1.amazonaws.com/dcrh9u79n7rtsa')

@app.route("/")
def index():
     # t = threading.Thread(target=func)
     # t.start()
     # t.join()
     # for thread in threading.enumerate(): 
     #      print(thread.name)
     return render_template("FinalVisualisation.html")

def func():
     print("__________Task 1 assigned to thread: {}".format(threading.current_thread().name)) 
     for thread in threading.enumerate(): 
          print(thread.name)
     # x=0
     # while True:
     #      if x>10000000:
     #          break
     #      if x% 100000 == 0:
     #           print("Log--------------%s" % (x))   
     #      x+=1


@app.route("/twitter_aug")
def test():
     return render_template("twitter_aug.html")


@app.route("/request_html/<year_month>")
def request_html(year_month):
     print("Retrieving html file from {}".format(year_month))
     # Connect to PostgreSQL server
     dbConnection = engine.connect()
     # Read data from PostgreSQL database table and load into a DataFrame instance
     dataFrame = pd.read_sql("select * from \"html_table\"", dbConnection)
     pd.set_option('display.expand_frame_repr', False)
     # Close the database connection
     dbConnection.close()
     html_str = dataFrame.iloc[0,0]
     print("requested_html")
     return html_str

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
