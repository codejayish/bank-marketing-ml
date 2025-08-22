# src/schema.py

TARGET_COL = "y"  # yes/no

# Bank Marketing (UCI "bank-additional-full.csv") expected columns
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome"
]

NUMERIC_COLS = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

ALL_COLS = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]

# For UI dropdowns (aligned with dataset)
CHOICES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school",
                  "illiterate", "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "day_of_week": ["mon","tue","wed","thu","fri"],
    "poutcome": ["failure","nonexistent","success"]
}
