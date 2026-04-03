import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("auto-mpg.csv")

# -------------------------------
# DATA CLEANING
# -------------------------------

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Drop missing values
df = df.dropna()

# Drop non-numeric column
df = df.drop(columns=['car name'])

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# One-hot encoding
df = pd.get_dummies(df, columns=['origin'], drop_first=True)

# New feature
df['power_to_weight'] = df['horsepower'] / df['weight']

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------

X = df.drop('mpg', axis=1)
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# PREDICTIONS
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# EVALUATION
# -------------------------------

print("R2 Score:", r2_score(y_test, y_pred))

# FIX for sklearn versions
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# -------------------------------
# VISUALIZATION (OPTIONAL)
# -------------------------------

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
y_pred_dt = dt.predict(X_test)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=7,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)



from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid,
                    cv=5,
                    scoring='r2')

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Linear R2:", r2_score(y_test, y_pred))
print("Decision Tree R2:", r2_score(y_test, y_pred_dt))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))