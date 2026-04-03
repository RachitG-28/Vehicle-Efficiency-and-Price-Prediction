import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# CONFIG
# -------------------------------
TARGET_COLUMN = "mpg"

# -------------------------------
# CLEANING FUNCTION
# -------------------------------
def clean_data(df):

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    df = df.dropna(subset=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)

    X = X.fillna(X.median(numeric_only=True))
    X = pd.get_dummies(X, drop_first=True)

    return X, y

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("auto-mpg.csv")

X, y = clean_data(df)

model_columns = X.columns

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# PREDICTIONS
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# EVALUATION
# -------------------------------
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)

# -------------------------------
# SAVE FILES
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("Model trained and saved successfully!")