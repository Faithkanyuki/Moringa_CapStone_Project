# Optimizing Healthcare Resource Allocation for Diabetic Patients in Kenya Using Machine Learning

![Project Banner](https://github.com/Faithkanyuki/Moringa_CapStone_Project/blob/main/diabetes.png)

---
# Introduction

Diabetes poses a growing burden on Kenya’s healthcare system, where limited resources must serve an increasing number of patients. 
Inefficient allocation of healthcare resources often leads to preventable complications, hospital readmissions, and increased costs. 
This project explores the use of machine learning techniques to analyze patient data and predict healthcare needs, enabling data-driven decision-making. 
By identifying high-risk diabetic patients early, the study aims to support targeted interventions, optimize resource utilization, and improve patient outcomes within Kenya’s constrained healthcare environment.




## Project Overview
Diabetes-related hospital readmissions place significant strain on Kenya’s healthcare system by increasing costs, overcrowding hospitals, and stretching already limited healthcare resources.

This project applies **machine learning techniques** to predict **30-day hospital readmission risk** among diabetic patients. The goal is to enable **early identification of high-risk patients**, support **targeted interventions**, and improve **healthcare resource allocation**.

The project follows the **CRISP-DM framework**, ensuring a structured, industry-aligned data science process while contextualizing insights for the Kenyan healthcare environment.

---

## Phase 1: Business Understanding
Unplanned readmissions are costly and often preventable. In Kenya’s resource-constrained healthcare system, missing a high-risk patient has greater consequences than flagging a low-risk one.

### Business Objective
Develop a predictive model that identifies **diabetic patients at high risk of readmission before discharge**, enabling:
- Targeted follow-up care
- Improved discharge planning
- Efficient use of hospital resources

### Success Metrics
- **Primary:** Recall ≥ 65%  
- **Secondary:** Precision ≥ 40%, AUC ≥ 0.70  

Recall is prioritized to minimize missed high-risk patients.

---

## Phase 2: Data Understanding
- **Dataset Size:** 101,766 hospital encounters  
- **Hospitals:** 130 healthcare facilities  
- **Target Variable:** Readmission within 30 days (binary)  

### Feature Categories
- Patient demographics  
- Diagnoses and procedures  
- Medications  
- Hospital utilization patterns  

### Key Data Challenges
- Severe class imbalance (~11% readmitted)
- Missing values in clinical variables
- High-cardinality categorical features

---

## Phase 3: Data Preparation & Feature Engineering
This phase transformed raw healthcare data into a model-ready dataset:
- Binary target creation for 30-day readmission
- Handling missing values and high-missing features
- Encoding categorical variables
- Feature engineering, including:
  - Hospital utilization indicators
  - Diagnosis groupings
  - Medication change flags
- Stratified train-test split to preserve class distribution

---

## Phase 4: Exploratory Data Analysis (EDA)
EDA was conducted to identify patterns and drivers of readmission risk.

Key findings showed higher readmission rates among patients with:
- Frequent hospital and emergency visits
- Longer hospital stays
- Certain diagnosis groupings
- Complex discharge dispositions

### Readmission rate by categorical variables
<img width="1139" height="1325" alt="image" src="https://github.com/user-attachments/assets/e6a2906f-df85-4f3a-8069-ebc4b279ef14" />

Readmission rates vary significantly across patient demographics, admission characteristics, discharge disposition, and diabetes severity.
Higher readmission risk is observed among older patients, emergency and urgent admissions, and patients discharged to post-acute care facilities.
Gender shows minimal influence, while clinical severity and missing diabetes tests are associated with increased readmissions. 
These patterns highlight the importance of using data-driven models to identify high-risk patients and support targeted interventions before discharge.

### Distributions of Numerical Variables by Readmission Status

<img width="1288" height="884" alt="image" src="https://github.com/user-attachments/assets/c2fee6bf-5fbb-4807-be77-8cd0fa18fa1c" />

Patients who were readmitted generally show longer hospital stays, higher numbers of lab procedures, and more medications prescribed compared to those not readmitted. 
Readmitted patients also tend to have more previous hospital visits, indicating higher overall care complexity. 
Changes in medications are slightly more frequent among readmitted patients, while emergency visit counts show minimal variation. 
Overall, higher clinical intensity and prior utilization are associated with increased readmission risk.

### Correlation Matrix of Numerical Features
<img width="1044" height="856" alt="image" src="https://github.com/user-attachments/assets/8ea20969-0f73-4d3d-93fe-6af3ac14be83" />

Most clinical variables show weak to moderate correlations, indicating low multicollinearity. 
Readmission within 30 days has no strong linear correlation with any single variable, confirming that readmission risk is influenced by a combination of factors. 
Strong correlations appear among utilization features (hospital visits, emergency and inpatient counts), supporting the need for a machine-learning approach rather than rule-based thresholds.

### Readmission Rate by Diagnosis Groups
This heatmap highlights diagnosis categories associated with higher readmission risk.

![Readmission Rate by Diagnosis Groups](https://github.com/user-attachments/assets/27bbfda3-d4c4-4854-98e4-2543b9930415)

---

Readmission rates increase with longer hospital stays, peaking for patients hospitalized 8–10 days, indicating higher clinical complexity and greater post-discharge risk
Readmission rates vary across diagnosis categories and diagnosis positions. Higher rates are observed in neoplasms, supplementary, 
and unknown diagnoses, particularly when appearing as primary diagnoses. This suggests that underlying disease complexity influences readmission risk and should be incorporated into predictive modeling.

### Readmission Rate by Time in Hospital
Longer hospital stays were associated with increased likelihood of readmission.

![Readmission Rate by Time in Hospital](https://github.com/user-attachments/assets/96534855-1304-42d8-afe9-ba79c4a4d16e)

---

## Phase 5: Modeling
Three machine learning models were trained and evaluated to predict 30-day hospital readmission among diabetic patients.

Models used:
1. **Logistic Regression** (baseline and SMOTE-enhanced)
2. **Random Forest**
3. **XGBoost**

Key modeling considerations included:
- Addressing class imbalance  
- Hyperparameter tuning  
- Threshold optimization to maximize Recall  

---

### 1️⃣ Logistic Regression

Logistic Regression was used as the baseline model to establish initial performance and provide interpretability.  
A SMOTE-enhanced version was also trained to address class imbalance and improve recall for readmitted patients.


<img width="1072" height="884" alt="image" src="https://github.com/user-attachments/assets/96f61431-d66c-427b-8052-1218532dd9ce" />

### Comprehensive visualizations for the improved model
<img width="1144" height="1031" alt="image" src="https://github.com/user-attachments/assets/f61e4a0a-581b-43fa-8302-f5d80a685f4f" />
### Threshold analysis visualization
<img width="1006" height="144" alt="image" src="https://github.com/user-attachments/assets/ee668683-7700-4a6e-809f-d669903d745f" />
### Final summary visualization
<img width="718" height="424" alt="image" src="https://github.com/user-attachments/assets/07571754-4be2-40c8-aa05-f9a00a54a4a7" />


---


### 2️⃣ Random Forest

Random Forest was implemented to capture non-linear relationships and interactions between patient features.  
Its ensemble structure improves robustness and reduces overfitting compared to linear models.

### Comparison with logistic 
<img width="1288" height="884" alt="image" src="https://github.com/user-attachments/assets/e0445427-8822-4dbf-8801-b88649448586" />


---

### 3️⃣ XGBoost

XGBoost was used as the final model due to its ability to iteratively correct prediction errors and optimize performance.  
It achieved the strongest overall results and was selected as the best-performing model.

<img width="1287" height="884" alt="image" src="https://github.com/user-attachments/assets/2f798b49-0961-481d-96f2-f6f49a99b147" />

---

## Phase 6: Model Evaluation & Comparison

### Performance Metrics

| Model | Recall | Precision | AUC |
|------|-------|-----------|-----|
| Logistic Regression | 0.651 | 0.131 | 0.588 |
| Random Forest | **0.690** | 0.154 | 0.660 |
| XGBoost | 0.655 | **0.163** | **0.672** |

---

### Model Performance Comparison
This visualization compares performance across all models under Kenya’s healthcare priorities.

![Model Performance Comparison](https://github.com/user-attachments/assets/239c5267-2904-4bb5-9f29-571c1731566d)

---

### Improved Logistic Regression – Business Impact
This visualization highlights the trade-off between Recall and false positives.

![Logistic Regression Business Impact](https://github.com/user-attachments/assets/5df7565a-8342-4341-8372-d106397e117d)

---

## Phase 7: Final Model Selection

### ✅ Recommended Model: **Random Forest**

**Rationale:**
- Achieved the **highest Recall (69%)**
- Maintained reasonable Precision
- Demonstrated strong generalization
- Easier to interpret and deploy compared to more complex models

This model best balances **patient safety**, **operational feasibility**, and **policy relevance**.

---

## Phase 8: Business Impact & Recommendations
- Enables early identification of high-risk diabetic patients
- Supports targeted discharge planning and follow-up care
- Improves hospital bed, staffing, and resource allocation
- Provides evidence-based insights for healthcare policy and planning

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Authors
FAITH KANYUKI

EDINAH OGOTI

CINDY AKINYI

DIANA ALOO

GODFREY OSUNDWA

ELSIE WAIRIMU

MITCHELLE MKAN

**Group Project – Moringa School**  

 Data Science  


