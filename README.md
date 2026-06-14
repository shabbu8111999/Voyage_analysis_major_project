# Voyage Analytics — Integrating MLOps in Travel

This is a group project for our MLOps course.
The goal is to build and deploy machine learning models on travel data — flights, users, and hotels.

I worked on **3 parts** of this project:
1. Data Analysis (Loading, Cleaning, EDA)
2. Regression Model — predicting flight prices
3. REST API — serving the model predictions via Flask

---

## What This Project Does

We have 3 datasets from a travel booking platform:
- **flights.csv** — 271,888 flight booking records
- **users.csv** — 1,340 traveller profiles
- **hotels.csv** — 40,552 hotel stay records

My part focuses on the flights and users data to build a model that **predicts the price of a flight** based on details like flight type, distance, agency, traveller age and more.

---

## My Contributions

### 1. Data Loading (`notebooks/01_data_loading.ipynb`)
- Loaded all 3 CSV files using pandas
- Checked the shape, column names and data types of each file
- Understood what each column means before touching any data

### 2. Data Cleaning (`notebooks/02_data_cleaning.ipynb`)
- Merged flights and users into one table using `userCode`
- Checked for missing values — none found
- Checked for duplicate rows — none found
- Fixed the `date` column from text to datetime format
- Renamed `from` → `origin` and `to` → `destination` to avoid Python keyword issues
- Dropped `travelCode` and `name` columns (not useful for prediction)
- Checked for outliers using IQR method on price, time and distance — none found
- Saved cleaned data to `data/processed/flights_merged.csv`

### 3. Univariate EDA (`notebooks/03_eda_univariate.ipynb`)
- Analysed one column at a time
- Histograms for: price, time, distance, age, hotel price
- Bar charts for: flightType, agency, gender, company
- Found that price ranges from 301 to 1754 BRL with peak around 900
- Found all categorical columns are well balanced

### 4. Bivariate EDA (`notebooks/04_eda_bivariate.ipynb`)
- Analysed two columns at a time using all 3 datasets
- Box plot — price vs flight type (first class is most expensive)
- Scatter plot — price vs distance (positive relationship)
- Line graph — monthly average price trend (no strong seasonal pattern)
- Bar chart — average price by gender (gender doesn't affect price much)
- Pie chart — hotel bookings by city (from hotels.csv)
- Bar chart — average hotel price by hotel name (from hotels.csv)

### 5. Multivariate EDA (`notebooks/05_eda_multivariate.ipynb`)
- Analysed 3 or more columns together using all 3 datasets
- Correlation heatmap — price and distance have strongest relationship
- Scatter with colour — price vs distance coloured by flight type
- Grouped bar — average price by flight type × agency
- Pair plot — price, time, distance coloured by flight type
- Stacked bar — gender split across flight types (flights + users data)
- Multi-line graph — monthly price trend per flight type
- Grouped bar — hotel total bill by city × hotel name (hotels data)

### 6. Regression Model (`notebooks/06_model_training.ipynb`)
- Target variable: `price` (continuous number → regression problem)
- Dropped useless columns: `userCode`, `origin`, `destination`
- Extracted date features: `month`, `day_of_week`, `is_weekend`, `quarter`
- Label encoded text columns: `flightType`, `agency`, `gender`, `company`
- Split data: 80% training, 20% testing
- Scaled numeric features using StandardScaler
- Trained **Linear Regression** as baseline
- Trained **Random Forest** as main model
- Checked for overfitting by comparing train R² vs test R²
- Checked feature importance — distance and flightType ranked highest
- Saved 3 files: `model.pkl`, `label_encoders.pkl`, `scaler.pkl`

### 7. REST API (`api/app.py`)
- Built using Flask
- Loads the 3 pkl files once when server starts
- **GET /health** — confirms server is running
- **GET /test** — opens a browser form to test predictions
- **POST /predict** — accepts JSON input, returns predicted flight price
- Input is validated before prediction — returns clear error if field is wrong or missing
- Dropdowns in the test form are built from saved label encoders automatically

---

## Tech Stack

| Tool | Used For |
|---|---|
| Python 3.11 | Main language |
| pandas | Data loading, cleaning, merging |
| numpy | Math operations |
| matplotlib | Plotting charts |
| seaborn | Advanced visualisations |
| scikit-learn | Label encoding, scaling, model training |
| joblib | Saving and loading pkl files |
| Flask | REST API |
| Jupyter Lab | Writing and running notebooks |
| uv | Package management |

---

## Folder Structure

```
Voyage_Project/
│
├── data/
│   ├── raw/
│   │   ├── flights.csv
│   │   ├── users.csv
│   │   └── hotels.csv
│   └── processed/
│       └── flights_merged.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_univariate.ipynb
│   ├── 04_eda_bivariate.ipynb
│   ├── 05_eda_multivariate.ipynb
│   └── 06_model_training.ipynb
│
├── models/
│   └── saved/
│       ├── model.pkl
│       ├── label_encoders.pkl
│       └── scaler.pkl
│
├── api/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## How to Set Up

```bash
# 1. Clone the project
git clone <your-repo-url>
cd Voyage_Project

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Add the 3 CSV files to data/raw/
#    flights.csv, users.csv, hotels.csv
```

---

## How to Run

Open and run each notebook from 01 to 06 in order.
After notebook 06 finishes, the `models/saved/` folder will have all 3 pkl files.

### Step 1 — Start the API
```bash
uv run python api/app.py
```

### Step 2 — Test in browser
```
http://localhost:5000/health   → check server is running
http://localhost:5000/test     → open the prediction form
```

---

## API — How to Predict

**Endpoint:** `POST /predict`

**Sample request body:**
```json
{
    "flightType"  : "firstClass",
    "agency"      : "FlyingDrops",
    "gender"      : "male",
    "company"     : "4You",
    "time"        : 1.76,
    "distance"    : 676.53,
    "age"         : 21,
    "month"       : 9,
    "day_of_week" : 3,
    "is_weekend"  : 0,
    "quarter"     : 3
}
```

**Response:**
```json
{
    "predicted_price": 1434.52,
    "currency": "BRL",
    "status": "success"
}
```

**Valid values for text fields:**
| Field | Valid Values |
|---|---|
| flightType | economic, premium, firstClass |
| agency | CloudFy, FlyingDrops, Rainbow |
| gender | male, female, none |
| company | values from your training data |

---

## Dataset Info

| File | Rows | Columns | Used For |
|---|---|---|---|
| flights.csv | 271,888 | 10 | Regression model (predict price) |
| users.csv | 1,340 | 5 | Merged with flights for age and gender features |
| hotels.csv | 40,552 | 8 | EDA only (bivariate + multivariate analysis) |
