from flask import Flask, render_template, request, jsonify
from BARTmodel import PredictionClass


app = Flask(__name__)

PredictionObj = PredictionClass()


@app.route("/", methods=["GET"])
def HomePage():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def PredictionRoute():
    if request.json is not None:
        text = request.json["text"]
        res = PredictionObj.Prediction(text)
        return jsonify(res)
    else:
        return jsonify({"Wrong Method": " please call by json request"})


if __name__ == "__main__":
    app.run()
