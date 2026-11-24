
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Create model folder
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 2. Load dataset
DATA_PATH = Path("data/Pune_House_Data.csv")
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 3. Convert area values into numeric square feet
def convert_area(value):
    if pd.isna(value):
        return np.nan

    value = str(value).replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", value)

    if not numbers:
        return np.nan

    numbers = [float(n) for n in numbers[:2]]
    return sum(numbers) / len(numbers)


df["total_sqft"] = df["total_sqft"].apply(convert_area)

# 4. Extract BHK from size
def extract_bhk(value):
    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


df["BHK"] = df["size"].apply(extract_bhk)

# 5. Select features and target
features = [
    "site_location",
    "area_type",
    "total_sqft",
    "BHK",
    "bath",
    "balcony"
]

target = "price"

df = df[features + [target]].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(
    subset=["site_location", "total_sqft", "BHK", target],
    inplace=True
)

df = df[
    (df["total_sqft"] > 0) &
    (df["BHK"] > 0) &
    (df["price"] > 0)
]

print("Cleaned dataset shape:", df.shape)

# 6. Split features and target
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Preprocessing
numeric_features = ["total_sqft", "BHK", "bath", "balcony"]
categorical_features = ["site_location", "area_type"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 8. Define models
models_to_compare = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

# 9. Train, evaluate, and compare models
results = []
trained_models = {}

for name, regressor in models_to_compare.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regressor", clone(regressor))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    trained_models[name] = pipeline

# 10. Display comparison
results_df = pd.DataFrame(results).sort_values(
    by="R2 Score",
    ascending=False
)

print("\nMODEL COMPARISON")
print("---------------------------")
print(results_df.to_string(index=False))

# 11. Save the best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(best_model, MODEL_DIR / "real_estate_model.pkl")

print("\nBest Model:", best_model_name)
print("Best R2 Score:", results_df.iloc[0]["R2 Score"])
print("Best model saved successfully!")

# 12. Generate model comparison graph
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.title("Machine Learning Model Comparison - R2 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Model comparison graph saved!")

# 13. Generate feature importance graph
# The selected best model is Random Forest for the current results.
# This section works with the fitted pipeline's actual step names.

preprocessor_fitted = best_model.named_steps["preprocessor"]
regressor_fitted = best_model.named_steps["regressor"]

feature_names = preprocessor_fitted.get_feature_names_out()
importance_values = regressor_fitted.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Save all feature importance values
importance_df.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)

# Plot the 15 most important features
top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Property Features")
plt.title("Top 15 Features Influencing Property Price")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Feature importance graph saved!")
print("Feature importance data saved!")
print("\nAll training and graph generation tasks completed successfully.")