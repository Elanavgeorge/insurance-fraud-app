# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv("Automobile_insurance_fraud.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================================================
# 3. BASIC DATA CLEANING
# =========================================================

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Fill missing categorical values
for col in ["collision_type", "property_damage", "police_report_available"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# =========================================================
# 4. TARGET VARIABLE ENCODING
# =========================================================

df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# =========================================================
# 5. VISUALIZATION 1: FRAUD DISTRIBUTION
# =========================================================

plt.figure(figsize=(6, 4))
sns.countplot(x="fraud_reported", data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.xlabel("Fraud Reported (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# =========================================================
# 6. VISUALIZATION 2: INCIDENT SEVERITY VS FRAUD
# =========================================================

plt.figure(figsize=(8, 4))
sns.countplot(x="incident_severity", hue="fraud_reported", data=df)
plt.title("Incident Severity vs Fraud")
plt.xticks(rotation=45)
plt.show()

# =========================================================
# 7. VISUALIZATION 3: VEHICLE CLAIM VS FRAUD
# =========================================================

plt.figure(figsize=(6, 4))
sns.boxplot(x="fraud_reported", y="vehicle_claim", data=df)
plt.title("Vehicle Claim Amount vs Fraud")
plt.show()

# =========================================================
# 8. VISUALIZATION 4: CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(10, 6))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# =========================================================
# 9. DROP UNNECESSARY COLUMNS
# =========================================================

drop_cols = [
    "policy_number",
    "policy_bind_date",
    "policy_state",
    "insured_zip",
    "incident_location",
    "incident_date",
    "incident_state",
    "incident_city",
    "insured_hobbies",
    "auto_make",
    "auto_model",
    "auto_year",
    "_c39"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

# =========================================================
# 10. FEATURE / TARGET SPLIT
# =========================================================

X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

# =========================================================
# 11. IDENTIFY NUMERIC & CATEGORICAL FEATURES
# =========================================================

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# =========================================================
# 12. PREPROCESSING PIPELINE (FIXED)
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ]
)

# =========================================================
# 13. MODEL PIPELINE
# =========================================================

# =========================================================
# 13. IMPROVED MODEL PIPELINE (ACCURACY BOOST)
# =========================================================

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=400,              # more trees → better learning
        max_depth=20,                  # controls overfitting
        min_samples_split=5,           # better generalization
        min_samples_leaf=2,            # avoids noisy splits
        max_features="sqrt",           # best for classification
        class_weight="balanced",       # handles fraud imbalance
        random_state=42,
        n_jobs=-1
    ))
])


# =========================================================
# 14. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# =========================================================
# 15. TRAIN MODEL
# =========================================================

pipeline.fit(X_train, y_train)

# =========================================================
# 16. MODEL EVALUATION
# =========================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# 17. CONFUSION MATRIX VISUALIZATION
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Fraud", "Fraud"],
    yticklabels=["Non-Fraud", "Fraud"]
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# =========================================================
# 18. SAVE TRAINED MODEL & FEATURE INFO
# =========================================================

joblib.dump(pipeline, "fraud_pipeline.pkl")

# Save original feature columns (for app input alignment)
joblib.dump(X.columns.tolist(), "features.pkl")

print("\n✅ Model saved as fraud_pipeline.pkl")
print("✅ Feature list saved as features.pkl")







