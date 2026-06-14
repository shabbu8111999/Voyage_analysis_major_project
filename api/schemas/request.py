# all fields the model needs to make a prediction
REQUIRED_FIELDS = [
    "flightType",   # text  : economic / premium / firstClass
    "agency",       # text  : CloudFy / FlyingDrops / Rainbow
    "gender",       # text  : male / female / none
    "company",      # text  : company name (4You, etc.)
    "time",         # float : flight duration in hours
    "distance",     # float : route distance in km
    "age",          # int   : traveller age
    "month",        # int   : 1 to 12
    "day_of_week",  # int   : 0=Monday to 6=Sunday
    "is_weekend",   # int   : 1 if Saturday or Sunday, else 0
    "quarter"       # int   : 1 to 4
]

# text fields — need label encoding before model can use them
TEXT_FIELDS = ["flightType", "agency", "gender", "company"]

# numeric fields — need scaling before model can use them
NUMERIC_FIELDS = ["time", "distance", "age", "month", "day_of_week"]

# exact column order the model was trained on
# input DataFrame must match this order
FEATURE_ORDER = [
    "flightType", "time", "distance", "agency",
    "company", "gender", "age", "month",
    "day_of_week", "is_weekend", "quarter"
]