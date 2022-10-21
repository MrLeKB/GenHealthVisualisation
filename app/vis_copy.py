from flask import Flask, render_template

app2 = Flask(__name__)

@app2.route("/")
def index():
     print("second thread______________________________________________")


