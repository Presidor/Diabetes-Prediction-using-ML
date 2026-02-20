# Diabetes-Prediction-using-ML

This repository contains a machine learning pipeline for predicting diabetes risk using the **Pima Indians Diabetes Dataset**. The project explores multiple classification algorithms, performs hyperparameter tuning, and evaluates model performance to identify individuals at risk of diabetes.
![Project Screenshot](https://github.com/Presidor/Diabetes-Prediction-using-ML/blob/main/Screenshot.png)

## 📊 Dataset
The dataset includes 768 entries with the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (binary: 1 = diabetes, 0 = no diabetes)

## 🔎 Project Workflow
1. **Data Loading & Cleaning**
   - Handle missing values and outliers
   - Remove invalid zero entries for critical features (Glucose, Insulin)

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Correlation heatmap
   - 3D visualization of BMI, Age, and Glucose

3. **Model Training**
   - Logistic Regression
   - Decision Trees
   - SVM
   - KNN
   - Random Forest
   - Gradient Boosting
   - XGBoost

4. **Hyperparameter Tuning**
   - GridSearchCV applied to Logistic Regression, Random Forest, and XGBoost
   - Best parameters identified for each model

5. **Evaluation**
   - Accuracy, F1-score, ROC-AUC
   - Confusion matrix and classification report

## ⚙️ Best Results
- Logistic Regression (tuned): ~78% accuracy
- Random Forest (tuned): ~77% accuracy
- XGBoost (tuned): ~74% accuracy

## 📦 Requirements
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
