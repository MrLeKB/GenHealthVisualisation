from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
     return render_template("FinalVisualisation.html")

@app.route("/twitter_aug")
def test():
     return render_template("twitter_aug.html")

@app.route("/test")
def test2():
     print("test")
     return None

if __name__ == "__main__":
     print('test')

