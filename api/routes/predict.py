import pandas as pd
from flask import Blueprint, request, jsonify, current_app

from api.schemas.request import (
    REQUIRED_FIELDS,
    TEXT_FIELDS,
    NUMERIC_FIELDS,
    FEATURE_ORDER
)


# creating the blueprint — "predict" is just its name
predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON with flight details, returns predicted price.

    Sample request body:
    {
        "flightType" : "firstClass",
        "agency"     : "FlyingDrops",
        "gender"     : "male",
        "company"    : "4You",
        "time"       : 1.76,
        "distance"   : 676.53,
        "age"        : 21,
        "month"      : 9,
        "day_of_week": 3,
        "is_weekend" : 0,
        "quarter"    : 3
    }
    """

    # reading JSON from the request body
    data = request.get_json()

    # checking if JSON was sent at all
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    # checking all required fields are present
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # getting the 3 pkl objects loaded in app.py
    model          = current_app.config["MODEL"]
    label_encoders = current_app.config["LABEL_ENCODERS"]
    scaler         = current_app.config["SCALER"]

    # building a one-row DataFrame from the request data
    row = {field: [data[field]] for field in REQUIRED_FIELDS}
    input_df = pd.DataFrame(row)

    # encoding text fields — same way as notebook 06
    for col in TEXT_FIELDS:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            valid = list(le.classes_)
            return jsonify({
                "error": f"Invalid value '{data[col]}' for '{col}'. Valid values: {valid}"
            }), 400

    # scaling numeric fields — same scaler from notebook 06
    input_df[NUMERIC_FIELDS] = scaler.transform(input_df[NUMERIC_FIELDS])

    # reordering columns to match training column order
    input_df = input_df[FEATURE_ORDER]

    # predicting price
    predicted_price = model.predict(input_df)[0]

    # returning result as JSON
    return jsonify({
        "predicted_price" : round(float(predicted_price), 2),
        "currency"        : "BRL",
        "status"          : "success"
    })