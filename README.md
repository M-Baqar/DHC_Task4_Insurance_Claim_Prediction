# Task 4: Predicting Insurance Claim Amounts

> **Internship:** Data Science & Analytics — DevelopersHub Corporation  
> **Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Google Colab

---

## Task Objective

Medical insurance companies need to accurately **estimate how much a customer will cost** in claims before setting their premium. Charging too little leads to losses; charging too much drives customers away.

The objective of this task is to build a **regression model** that predicts the **medical insurance charges** billed to an individual based on personal attributes such as age, BMI, smoking status, and region.

**Target Variable:** `charges` — the medical insurance cost billed (continuous numeric value in USD)

---

## Approach

### 1. Data Loading
- Loaded the Medical Cost Personal Dataset from a public GitHub mirror — **no Kaggle account required**, auto-downloads in the notebook
- Dataset contains 1,338 rows and 7 columns

### 2. Data Cleaning & Preparation
- Confirmed **no missing values** in the dataset
- Applied **Label Encoding** to binary categorical columns:
  - `sex`: female=0, male=1
  - `smoker`: no=0, yes=1
- Applied **One-Hot Encoding** to `region` (4 categories: northeast, northwest, southeast, southwest)

### 3. Exploratory Data Analysis (EDA)
Visualizations focused on the 3 key features requested by the task:

| Chart | Feature Analyzed | Insight Sought |
|-------|-----------------|---------------|
| Histogram + KDE | charges | Overall distribution of insurance costs |
| Box plot + Violin plot | smoker vs charges | How much does smoking increase charges? |
| Scatter plot | BMI vs charges (by smoker) | Combined impact of BMI and smoking |
| Scatter plot | Age vs charges (by smoker) | How age and smoking interact on cost |
| Box plot | children vs charges | Effect of family size on cost |
| Correlation Heatmap | All features | Feature-to-feature relationships |

### 4. Model Training
- Split data: **80% training / 20% testing**
- Trained two models for comparison:
  - **Linear Regression** — baseline model, interpretable
  - **Random Forest Regressor** (100 trees) — captures non-linear relationships

### 5. Model Evaluation
Both models evaluated using three regression metrics:
- **MAE (Mean Absolute Error)** — average dollar error in predictions
- **RMSE (Root Mean Squared Error)** — penalizes large errors more heavily
- **R² (R-squared)** — proportion of variance explained by the model (higher = better)

Additional visualizations:
- Actual vs Predicted scatter plots for both models
- Side-by-side MAE, RMSE, and R² comparison bar charts
- Residual plot to check for prediction bias
- Feature importance chart from Random Forest

---

## Results & Insights

### Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | ~$4,200 | ~$6,100 | ~0.75 |
| Random Forest Regressor | ~$2,500 | ~$4,600 | ~0.87 |

Random Forest significantly outperformed Linear Regression because insurance charges have **non-linear relationships** with features (especially the smoking+BMI interaction).

### Key Findings

1. **Smoking is the single most powerful predictor** — smokers pay on average **3 to 4 times more** in insurance charges than non-smokers with identical profiles. This is the most important business finding.

2. **BMI and smoking interact dramatically** — smokers with a BMI above 30 (classified as obese) have charges that are exponentially higher than obese non-smokers. The scatter plots show a clear two-cluster pattern divided by smoking status.

3. **Age has a steady positive effect** — older individuals pay more, as expected. The relationship appears roughly linear within each smoking group.

4. **Region has minimal impact** — geography (northeast, northwest, southeast, southwest) contributes very little to the prediction, suggesting fairly uniform insurance pricing across US regions.

5. **Number of children** — having 0–3 children shows relatively stable charges. Having 4–5 children shows slightly higher costs, though the sample size is small.

6. **Linear Regression limitations** — the residual plot showed that Linear Regression struggles with high-charge predictions (for smokers), where it consistently underestimates. This confirms a non-linear relationship that requires tree-based models.

### Business Recommendations
- **Smoking surcharge** should be the biggest factor in premium pricing — the data fully supports this
- Customers with **high BMI + smoking** represent the highest-cost segment and should be charged accordingly or incentivized to quit smoking
- Offering **smoking cessation programs** could significantly reduce claim amounts for the insurance company
- A Random Forest model should be used in production over Linear Regression given the R² improvement of ~12%

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Medical Cost Personal Dataset |
| Kaggle Link | https://www.kaggle.com/datasets/mirichoi0218/insurance |
| Public Mirror | https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv |
| Rows | 1,338 |
| Columns | 7 (6 features + 1 target) |
| Target | `charges` (continuous USD value) |
| Missing Values | None |
