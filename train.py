import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

print("1. Loading California Housing Dataset...")
# Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
data = fetch_california_housing()
X = data.data
y_price = data.target  # Note: Target is in units of $100,000

# Create a classification target: 1 if price > median statewide price, 0 otherwise
median_price = np.median(y_price)
y_class = (y_price > median_price).astype(int)

print("2. Splitting data into 80% Training and 20% Testing...")
# We split both targets simultaneously using the exact same random seed
X_train, X_test, y_price_train, y_price_test, y_class_train, y_class_test = train_test_split(
    X, y_price, y_class, test_size=0.2, random_state=42
)

# --- Train & Evaluate Regressor ---
print("\n3. Training Regressor (This takes ~5-10 seconds)...")
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_price_train)

reg_preds = reg_model.predict(X_test)
print("\n--- Regressor Performance ---")
print(f"R² Score (Closer to 1.0 is better): {r2_score(y_price_test, reg_preds):.4f}")
print(f"Mean Error: ${(mean_absolute_error(y_price_test, reg_preds) * 100000):,.2f} off on average")
joblib.dump(reg_model, "regressor_model.joblib")

# --- Train & Evaluate Classifier ---
print("\n4. Training Classifier (This takes ~5-10 seconds)...")
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_class_train)

clf_preds = clf_model.predict(X_test)
print("\n--- Classifier Performance ---")
print(f"Accuracy: {accuracy_score(y_class_test, clf_preds) * 100:.2f}%")
print(classification_report(y_class_test, clf_preds, target_names=["Below Median", "Above Median"]))
joblib.dump(clf_model, "classifier_model.joblib")

print("\nModels successfully evaluated and saved!")