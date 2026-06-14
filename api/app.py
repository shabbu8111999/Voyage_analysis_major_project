import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
 
# adding project root to path so pkl files can be found
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
 
 
# 1. Create Flask app
 
app = Flask(__name__)
 
 
# 2. Load the 3 pkl files once when server starts
 
SAVED_DIR = ROOT / "models" / "saved"
 
print("Loading model...")
model = joblib.load(SAVED_DIR / "model.pkl")
 
print("Loading label encoders...")
label_encoders = joblib.load(SAVED_DIR / "label_encoders.pkl")
 
print("Loading scaler...")
scaler = joblib.load(SAVED_DIR / "scaler.pkl")
 
print("All files loaded. Server is ready.\n")
 
 
# 3. Field definitions
 
REQUIRED_FIELDS = [
    "flightType", "agency", "gender", "company",
    "time", "distance", "age",
    "month", "day_of_week", "is_weekend", "quarter"
]
 
TEXT_FIELDS    = ["flightType", "agency", "gender", "company"]
NUMERIC_FIELDS = ["time", "distance", "age", "month", "day_of_week"]
 
FEATURE_ORDER = [
    "flightType", "time", "distance", "agency",
    "company", "gender", "age", "month",
    "day_of_week", "is_weekend", "quarter"
]
 
 
# 4. Home route
 
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "api"      : "Voyage Analytics — Flight Price Predictor",
        "version"  : "1.0",
        "endpoints": {
            "health" : "GET  /health",
            "test"   : "GET  /test   ← open this in browser to test",
            "predict": "POST /predict"
        }
    })
 
 
# 5. Health check route
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "ok",
        "message": "API is running"
    })
 
 
# 6. Browser test form route
 
@app.route("/test", methods=["GET"])
def test_page():
 
    # reading valid values directly from the saved label encoders
    # so the dropdowns always match what the model was trained on
    flight_types = list(label_encoders["flightType"].classes_)
    agencies     = list(label_encoders["agency"].classes_)
    genders      = list(label_encoders["gender"].classes_)
    companies    = list(label_encoders["company"].classes_)
 
    # building dropdown option HTML for each category
    ft_options  = "".join(f'<option value="{v}">{v}</option>' for v in flight_types)
    ag_options  = "".join(f'<option value="{v}">{v}</option>' for v in agencies)
    gen_options = "".join(f'<option value="{v}">{v}</option>' for v in genders)
    co_options  = "".join(f'<option value="{v}">{v}</option>' for v in companies)
 
    # returning the full HTML page as a string
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voyage Analytics — Predict Flight Price</title>
        <style>
            body       {{ font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
            h1         {{ color: #2c3e50; }}
            .card      {{ background: white; padding: 24px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            label      {{ display: block; margin-top: 14px; font-weight: bold; font-size: 13px; color: #333; }}
            input, select {{ width: 100%; padding: 9px; margin-top: 4px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; font-size: 14px; }}
            .row       {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
            button     {{ margin-top: 22px; width: 100%; padding: 13px; background: #2c3e50; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }}
            button:hover {{ background: #3d566e; }}
            #result    {{ margin-top: 20px; padding: 18px; border-radius: 8px; display: none; text-align: center; }}
            .success   {{ background: #d4edda; border: 1px solid #28a745; color: #155724; }}
            .error     {{ background: #f8d7da; border: 1px solid #dc3545; color: #721c24; }}
            .price     {{ font-size: 36px; font-weight: bold; margin: 8px 0; }}
            .subtitle  {{ font-size: 13px; color: #666; margin-top: 6px; }}
        </style>
    </head>
    <body>
        <h1>✈️ Flight Price Predictor</h1>
        <p>Fill in the flight details below and click <b>Predict Price</b>.</p>
 
        <div class="card">
 
            <div class="row">
                <div>
                    <label>Flight Type</label>
                    <select id="flightType">{ft_options}</select>
                </div>
                <div>
                    <label>Agency</label>
                    <select id="agency">{ag_options}</select>
                </div>
            </div>
 
            <div class="row">
                <div>
                    <label>Gender</label>
                    <select id="gender">{gen_options}</select>
                </div>
                <div>
                    <label>Company</label>
                    <select id="company">{co_options}</select>
                </div>
            </div>
 
            <div class="row">
                <div>
                    <label>Flight Duration (hours)</label>
                    <input type="number" id="time" value="1.76" step="0.1" min="0" />
                </div>
                <div>
                    <label>Distance (km)</label>
                    <input type="number" id="distance" value="676" step="1" min="0" />
                </div>
            </div>
 
            <div class="row">
                <div>
                    <label>Traveller Age</label>
                    <input type="number" id="age" value="25" min="1" max="100" />
                </div>
                <div>
                    <label>Month (1 - 12)</label>
                    <input type="number" id="month" value="6" min="1" max="12" />
                </div>
            </div>
 
            <div class="row">
                <div>
                    <label>Day of Week (0=Mon, 6=Sun)</label>
                    <input type="number" id="day_of_week" value="2" min="0" max="6" />
                </div>
                <div>
                    <label>Quarter (1 - 4)</label>
                    <input type="number" id="quarter" value="2" min="1" max="4" />
                </div>
            </div>
 
            <label>Is Weekend?</label>
            <select id="is_weekend">
                <option value="0">No — Weekday</option>
                <option value="1">Yes — Weekend</option>
            </select>
 
            <button onclick="predict()">🔍 Predict Price</button>
        </div>
 
        <div id="result"></div>
 
        <script>
            async function predict() {{
                // collecting values from all form fields
                const payload = {{
                    flightType  : document.getElementById("flightType").value,
                    agency      : document.getElementById("agency").value,
                    gender      : document.getElementById("gender").value,
                    company     : document.getElementById("company").value,
                    time        : parseFloat(document.getElementById("time").value),
                    distance    : parseFloat(document.getElementById("distance").value),
                    age         : parseInt(document.getElementById("age").value),
                    month       : parseInt(document.getElementById("month").value),
                    day_of_week : parseInt(document.getElementById("day_of_week").value),
                    is_weekend  : parseInt(document.getElementById("is_weekend").value),
                    quarter     : parseInt(document.getElementById("quarter").value)
                }};
 
                // sending POST request to /predict
                const res  = await fetch("/predict", {{
                    method  : "POST",
                    headers : {{ "Content-Type": "application/json" }},
                    body    : JSON.stringify(payload)
                }});
 
                const data = await res.json();
                const div  = document.getElementById("result");
                div.style.display = "block";
 
                if (res.ok) {{
                    div.className = "success";
                    div.innerHTML = `
                        <p>✅ Prediction Successful</p>
                        <p class="price">💰 ${{data.predicted_price}} ${{data.currency}}</p>
                        <p class="subtitle">
                            ${{payload.flightType}} &nbsp;|&nbsp;
                            ${{payload.agency}} &nbsp;|&nbsp;
                            ${{payload.distance}} km &nbsp;|&nbsp;
                            ${{payload.time}} hrs
                        </p>
                    `;
                }} else {{
                    div.className = "error";
                    div.innerHTML = `<p>❌ Error: ${{data.error}}</p>`;
                }}
            }}
        </script>
    </body>
    </html>
    """
 
 
# 7. Predict route
 
@app.route("/predict", methods=["POST"])
def predict():
 
    data = request.get_json()
 
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
 
    # checking all required fields are present
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
 
    # building one-row dataframe from request
    row      = {field: [data[field]] for field in REQUIRED_FIELDS}
    input_df = pd.DataFrame(row)
 
    # encoding text columns
    for col in TEXT_FIELDS:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            valid = list(le.classes_)
            return jsonify({
                "error": f"Invalid value '{data[col]}' for '{col}'. Valid: {valid}"
            }), 400
 
    # scaling numeric columns
    input_df[NUMERIC_FIELDS] = scaler.transform(input_df[NUMERIC_FIELDS])
 
    # reordering to match training column order
    input_df = input_df[FEATURE_ORDER]
 
    # predicting
    predicted_price = model.predict(input_df)[0]
 
    return jsonify({
        "predicted_price": round(float(predicted_price), 2),
        "currency"       : "BRL",
        "status"         : "success"
    })
 
 
# 8. Run the server
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open in browser → http://localhost:{port}/test")
    app.run(host="0.0.0.0", port=port, debug=True)