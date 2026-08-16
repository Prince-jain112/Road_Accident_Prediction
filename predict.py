import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Load encoders
encoders = joblib.load("encoders.pkl")

print("✅ Model Loaded")
print("✅ Encoders Loaded")

print("\nFeatures:")
print(model.feature_names_in_)

print("\nEncoded Cities:")
print(encoders["city"].classes_)

print("\nEncoded Weather:")
print(encoders["weather"].classes_)

