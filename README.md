# CODDSOFT-Task-1
# 🚢 Titanic Survival Prediction - CODSOFT Internship Task 1

This project is part of the *CODSOFT Data Science Internship*. The task involves building a machine learning model using the Titanic dataset to predict whether a passenger survived or not.

---

## 📌 Objective

To analyze passenger data from the Titanic disaster and build a predictive model that can determine survival outcomes based on features like age, sex, passenger class, etc.

---

## 🧠 Dataset

Dataset used: [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

- train.csv: Used for training the model (includes survival labels).
- test.csv: Used for testing and making final predictions.

---

## 🛠 Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 🔍 Steps Followed

1. *Data Loading & Exploration*  
   Loaded train.csv and test.csv, inspected data structure, and checked for missing values.

2. *Data Cleaning*  
   - Filled missing Age and Fare values with median.  
   - Filled Embarked with mode.  
   - Dropped Cabin due to many null values.

3. *Feature Engineering*  
   - Extracted passenger Title from Name.  
   - Encoded categorical columns (Sex, Embarked, Title).  
   - Dropped unnecessary columns like Ticket, Name.

4. *Model Building*  
   Trained two models:
   - Logistic Regression
   - Random Forest Classifier

5. *Model Evaluation*  
   - Evaluated on validation split from training data.  
   - Printed accuracy and classification report.

6. *Test Prediction*  
   - Predicted survival on test.csv.  
   - Exported submission.csv file.

---

## ✅ Results

Both models performed well, with *Random Forest* achieving higher accuracy on validation data. Final predictions were saved for submission.

---

## 🚀 How to Run

1. Open the notebook: Titanic_Survival_Prediction.ipynb
2. Upload the train.csv and test.csv when prompted (or place them in the same directory)
3. Run all cells to:
   - Train the model
   - Evaluate accuracy
   - Generate submission CSV

---

## 🔗 Colab Access

If you prefer running this on Google Colab, open with:

[[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hari03750/CODSOFT-Task-1/blob/main/Titanic_Survival_Prediction.ipynb)](https://colab.research.google.com/github/Hari03750/CODSOFT-Task-1/blob/main/Titanic_Survival_Prediction.ipynb
)



---

## 📌 Intern Information

This project is part of *CODSOFT Internship Task 1 - Data Science Track*  
> #codsoft #internship #titanic #datascience

---
