# Vehicle-Efficiency-and-Price-Prediction
# 🚗 Vehicle Analytics ML Pipeline

An end-to-end Machine Learning project to predict vehicle fuel efficiency (MPG) and car prices using regression models, with deployment via Streamlit.

---

## 📌 Overview

This project demonstrates a complete machine learning workflow:

* Data cleaning and preprocessing
* Feature engineering
* Model training and evaluation
* Model comparison (Linear, Decision Tree, Random Forest)
* Deployment using Streamlit

---

## 🎯 Objectives

* Predict vehicle fuel efficiency (MPG)
* Predict car prices
* Build a reusable ML pipeline for tabular datasets
* Deploy model as an interactive web application

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Matplotlib, Seaborn

---

## 📂 Project Structure

```
Vehicle-Analytics-ML-Pipeline/
│
├── data/
│   ├── auto-mpg.csv
│   ├── Carprice_Assignment.csv
│
├── model.py          # Model training & saving
├── app.py            # Streamlit web app
├── model.pkl         # Trained model
├── columns.pkl       # Feature columns
├── README.md
```

---

## ⚙️ Features

* Generic data preprocessing pipeline
* Automatic handling of categorical variables
* Feature engineering (e.g., power-to-weight ratio)
* Multiple model comparison
* Hyperparameter tuning (GridSearchCV)
* Interactive Streamlit UI for predictions

---

## 📊 Model Evaluation

The model performance is evaluated using:

* **R² Score** → Measures model accuracy
* **RMSE (Root Mean Squared Error)** → Measures prediction error
* **MAE (Mean Absolute Error)** → Measures average deviation

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

### 2️⃣ Train the Model

```
python model.py
```

### 3️⃣ Run the App

```
python -m streamlit run app.py
```

---

## 🌐 App Functionality

* Input vehicle details (horsepower, weight, cylinders, etc.)
* Predict fuel efficiency or car price
* Real-time results using trained ML model

---

## 🧠 Key Learnings

* Importance of data preprocessing consistency
* Impact of feature engineering on model performance
* Benefits of ensemble models (Random Forest)
* End-to-end ML pipeline development
* Deployment using Streamlit

---

## 🔮 Future Improvements

* Add model explainability (SHAP)
* Deploy on cloud (Render / Hugging Face)
* Improve UI/UX
* Add more datasets for generalization

---

## 👨‍💻 Author

**Rachit Goyal**
B.Tech CSE | Aspiring Data Analyst & ADAS Engineer

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
