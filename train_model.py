from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Clean Dataset Load
df = pd.read_csv("clean_dataset.csv")
print("Dataset Loaded Successfully")

# Label Encoder Object
le = LabelEncoder()
# Date Convert
df["date"] = pd.to_datetime(df["date"])

# New Features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Remove Original Date
df.drop("date", axis=1, inplace=True)
# Text Columns Find Karna
text_columns = df.select_dtypes(include=["object", "string"]).columns

print("\nText Columns:")
print(text_columns)


print("\n========== DATASET COLUMNS ==========")
print(df.columns.tolist())

print("\n========== SAMPLE DATA ==========")
print(df[
    [
        "cause",
        "casualties",
        "risk_score",
        "accident_severity"
    ]
].head(10))
# Encoding
# Dictionary to store encoders
print("\n========== UNIQUE VALUES ==========")

print("\nCities in Dataset:")
print(df["city"].unique())
print("Total Cities:", df["city"].nunique())

print("\nWeather:")
print(df["weather"].unique())

print("\nRoad Type:")
print(df["road_type"].unique())

print("\nVisibility:")
print(df["visibility"].unique())

print("\nTraffic Density:")
print(df["traffic_density"].unique())

print("\nState:")
print(df["state"].unique())

print("\nDay Of Week:")
print(df["day_of_week"].unique())
# Dictionary to store encoders
encoders = {}

# Encoding
for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nEncoding Completed Successfully")

print("\nAccident Severity Labels:")
print(encoders["accident_severity"].classes_)

print(df.head())

# Features and Target

X = df[
    [
        "city",
        "state",
        "road_type",
        "weather",
        "visibility",
        "traffic_density",
        "temperature",
        "hour",
        "day_of_week",
        "is_weekend",
        "traffic_signal",
        "vehicles_involved",
        "is_peak_hour"
    ]
]

Y = df["accident_severity"]

print("\n========== ACCIDENT SEVERITY DISTRIBUTION ==========")
print(df["accident_severity"].value_counts())
print(df["accident_severity"].value_counts(normalize=True) * 100)

print("\nFeatures Shape :", X.shape)
print("Target Shape :", Y.shape)



X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape :", X_train.shape)
print("Test Shape :", X_test.shape)

# Decision Tree Model

model = DecisionTreeClassifier(random_state=42)

# Model Training
model.fit(X_train, Y_train)

print("\n✅ Decision Tree Model Trained Successfully")

# Prediction

Y_pred = model.predict(X_test)

print("\nPrediction Completed")

accuracy = accuracy_score(Y_test, Y_pred)

print(f"\nDecision Tree Accuracy : {accuracy*100:.2f}%")

# Random Forest Model

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Training
rf_model.fit(X_train, Y_train)

print("\n✅ Random Forest Trained Successfully")

print("\n========== FEATURE IMPORTANCE ==========")

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

print(feature_importance.sort_values(ascending=False))

# Prediction
rf_pred = rf_model.predict(X_test)
print("\n========== RANDOM FOREST CLASSIFICATION REPORT ==========")

print(classification_report(
    Y_test,
    rf_pred,
    labels=[0, 1, 2],
    target_names=["Fatal", "Major", "Minor"]
))

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(
    Y_test,
    rf_pred,
    labels=[0, 1, 2]
))

# Accuracy
rf_accuracy = accuracy_score(Y_test, rf_pred)

print(f"Random Forest Accuracy : {rf_accuracy*100:.2f}%")

# Logistic Regression Model

lr_model = LogisticRegression(max_iter=1000)

# Training
lr_model.fit(X_train, Y_train)

print("\n✅ Logistic Regression Trained Successfully")

# Prediction
lr_pred = lr_model.predict(X_test)

print("\n========== LOGISTIC REGRESSION CLASSIFICATION REPORT ==========")

print(classification_report(
    Y_test,
    lr_pred,
    labels=[0, 1, 2],
    target_names=["Fatal", "Major", "Minor"]
))

print("\n========== LOGISTIC REGRESSION CONFUSION MATRIX ==========")

print(confusion_matrix(
    Y_test,
    lr_pred,
    labels=[0, 1, 2]
))

# Accuracy
lr_accuracy = accuracy_score(Y_test, lr_pred)

print(f"Logistic Regression Accuracy : {lr_accuracy*100:.2f}%")

# ==========================
# Naive Bayes Model
# ==========================

nb_model = GaussianNB()

# Training
nb_model.fit(X_train, Y_train)

print("\n✅ Naive Bayes Trained Successfully")

# Prediction
nb_pred = nb_model.predict(X_test)

# Accuracy
nb_accuracy = accuracy_score(Y_test, nb_pred)

print(f"Naive Bayes Accuracy : {nb_accuracy*100:.2f}%")

# ==========================
# Model Comparison
# ==========================

results = {
    "Decision Tree": accuracy,
    "Random Forest": rf_accuracy,
    "Logistic Regression": lr_accuracy,
    "Naive Bayes": nb_accuracy
}

print("\n========== MODEL COMPARISON ==========")

for model, score in results.items():
    print(f"{model} : {score*100:.2f}%")

best_model = max(results, key=results.get)

print("\n🏆 Best Model :", best_model)
print(f"Accuracy : {results[best_model]*100:.2f}%")

# Save Random Forest Model

joblib.dump(rf_model, "random_forest_model.pkl")

print("\n✅ Random Forest Model Saved Successfully")


# Accuracy
rf_accuracy = accuracy_score(Y_test, rf_pred)

print(f"Random Forest Accuracy : {rf_accuracy*100:.2f}%")

# Save Model
joblib.dump(rf_model, "random_forest_model.pkl")

print("\n✅ Random Forest Model Saved Successfully")

# Save all encoders
joblib.dump(encoders, "encoders.pkl")

print("✅ Encoders Saved Successfully")