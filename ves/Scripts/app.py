"""
Web Application Development 
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ["POST"])
def predict_api(): 
    data = request.json["data"]


if __name__ == '__main__':
    app.run(debug=True)