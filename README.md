# Optimizing Healthcare Resource Allocation for Diabetic Patients in Kenya Using Machine Learning

![Project Banner](https://github.com/Faithkanyuki/Moringa_CapStone_Project/blob/main/diabetes.png)

---

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

### Readmission Rate by Diagnosis Groups
This heatmap highlights diagnosis categories associated with higher readmission risk.

![Readmission Rate by Diagnosis Groups](https://github.com/user-attachments/assets/27bbfda3-d4c4-4854-98e4-2543b9930415)

---

### Readmission Rate by Time in Hospital
Longer hospital stays were associated with increased likelihood of readmission.

![Readmission Rate by Time in Hospital](https://github.com/user-attachments/assets/96534855-1304-42d8-afe9-ba79c4a4d16e)

---

## Phase 5: Modeling
Three machine learning models were trained and evaluated:
1. **Logistic Regression** (baseline and SMOTE-enhanced)
2. **Random Forest**
3. **XGBoost**

Key modeling considerations included:
- Addressing class imbalance
- Hyperparameter tuning
- Threshold optimization to maximize Recall

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
GEOFREY OSUNDWA
ELSIE WAIRIMU
MITCHEL MKAN
**Group Project – Moringa School**  
 Data Science  


