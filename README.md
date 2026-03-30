# 🌍 Life Expectancy Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting **life expectancy** based on country-level health, economic, and demographic indicators using machine learning techniques.

The goal is to build an **accurate, generalizable model** that can estimate life expectancy for unseen countries by leveraging a full ML pipeline including data preprocessing, feature engineering, model training, and evaluation.

---

## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)** to understand dataset patterns
* Handle **missing values and outliers** effectively
* Apply **feature engineering and transformation techniques**
* Train and evaluate multiple machine learning models
* Select the **best model based on performance and efficiency**
* Generate predictions on unseen test data

---

## 📊 Dataset

* **Size:** 2,071 rows × 23 columns
* **Type:** Country-year level data
* **Target Variable:** `Life Expectancy`

### Key Features

* Health indicators (BMI, mortality, diseases)
* Economic indicators (GDP, expenditure)
* Social indicators (schooling, income composition)

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Removed duplicates and handled missing values
* Country-wise analysis for better time-series consistency
* Imputation techniques:

  * **MultiOutputRegressor** (for correlated features)
  * **KNN Imputer**
  * Mode filling for categorical variables

---

### 2. Exploratory Data Analysis (EDA)

* Identified strong correlations:

  * Negative: Adult Mortality, HIV/AIDS
  * Positive: Schooling, BMI, Income
* Removed redundant features to avoid multicollinearity
* Designed feature combinations to improve representation

---

### 3. Feature Engineering

* Created new features:

  * `VaccineScore` = Polio + Diphtheria
  * `ThinnessIndex`
  * `EarlyMortality`
* Applied:

  * Log transformation
  * Standard scaling
* Train-test split based on **country groups** (important for generalization)

---

### 4. Model Training

Tested multiple models:

#### Tree-based Models (Primary)

* Gradient Boosting
* Hist Gradient Boosting
* Random Forest
* XGBoost
* LightGBM

#### Other Models (Baseline)

* Linear Regression / Ridge / ElasticNet
* KNN

Used:

* **GroupKFold Cross Validation (by country)**
* **RandomizedSearchCV** for hyperparameter tuning

---

## 📈 Results

| Model                | R² Score  | Notes              |
| -------------------- | --------- | ------------------ |
| **XGBoost**          | **0.893** | Best accuracy      |
| HistGradientBoosting | 0.875     | Best efficiency    |
| Random Forest        | ~0.87     | Strong performance |
| KNN                  | ~0.81     | Moderate           |
| Linear Models        | ~0.75     | Underfitting       |

---

## 🏆 Final Model

**Hist Gradient Boosting** was selected as the final model because:

* Competitive accuracy (R² ≈ 0.875)
* Much faster training time (~2× faster than XGBoost)
* Strong generalization on unseen countries

---

## ⚠️ Limitations

* Slightly lower accuracy than XGBoost
* Limited interpretability compared to linear models

### Future Improvements

* Use **SHAP** for model interpretability
* Apply more exhaustive hyperparameter tuning
* Explore deep learning approaches

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/life-expectancy-ml.git
cd life-expectancy-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook
```

Open the main notebook and run all cells.

---

## 📁 Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   └── main.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
├── results/
│   └── predictions.csv
├── README.md
└── requirements.txt
```

---

## 📌 Key Highlights

* Country-wise preprocessing strategy improves realism
* Group-based cross-validation prevents data leakage
* Tree-based models outperform linear models for this task
* Balanced trade-off between **accuracy and efficiency**

---

## 📜 Acknowledgement

This project was completed as part of the **COSC2753 - Machine Learning course** at RMIT University.

Assignment requirements included:

* Applying full ML pipeline
* Evaluating multiple models
* Justifying the final model selection
* Generating predictions for unseen data 
