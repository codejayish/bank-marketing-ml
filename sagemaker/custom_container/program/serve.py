# custom_container/program/serve.py
import os
import flask
from model_utils import load_model, input_fn, predict_fn, output_fn

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

app = flask.Flask(__name__)
model = None


@app.route("/ping", methods=["GET"])
def ping():
    """Check container health"""
    health = model is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Handle prediction requests"""
    global model
    if model is None:
        model = load_model(MODEL_DIR)

    data = flask.request.data.decode("utf-8")
    content_type = flask.request.content_type

    input_data = input_fn(data, content_type)
    predictions = predict_fn(input_data, model)

    accept = flask.request.headers.get("Accept", "application/json")
    response = output_fn(predictions, accept)
    return flask.Response(
        response=str(response),
        status=200,
        mimetype=accept
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
