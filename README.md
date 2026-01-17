# PROJECT TITLE: Optimizing Healthcare Resource Allocation for Diabetic Patients in Kenya Using Machine Learning

## PHASE 1: BUSINESS UNDERSTANDING

### Project Overview
Kenya's public health system faces significant strain due to chronic disease management. This project develops a predictive model to identify diabetic patients at high risk of 30-day hospital readmission

### Objectives
Develop a predictive machine learning model that identifies diabetic patients at high risk of 30-day hospital readmission before discharge. This will enable:
* Targeted interventions
* Optimized resource allocation
* Improved patient outcomes within Kenya's constrained healthcare system

### Key Questions¶
1. **Clinical Risk Factors**: Which patient characteristics and clinical factors are most predictive of 30-day readmission for diabetic patients in Kenya?
2. **Resource Optimization**: How can we best allocate limited healthcare resources (beds, specialists, follow-up care) to maximize impact on readmission rates?
3. **Intervention Effectiveness**: What targeted interventions (discharge planning, medication management, follow-up protocols) would be most effective for high-risk patients?
4. **Model Performance**: Can we achieve sufficient predictive accuracy (especially recall) to justify implementation in resource-constrained settings?

 ### Business Impact
- **Resource Prioritization:** Focus scarce resources (dedicated nurses, home-care kits, subsidized transportation) on highest-risk patients, improving efficiency by 40–60%  
- **Improved Patient Care:** Customized discharge plans for high-risk patients may reduce readmissions by 20–30%  
- **Reduced Hospital Strain:** Lower readmission rates, freeing hospital beds and staff for other critical needs  
- **Cost Reduction:** Decrease Ministry of Health expenditures on preventable readmissions (estimated 15–25% savings on diabetic care costs)  
- **Policy Informed Decision Making:** Provide data-driven insights for healthcare policy formulation and resource allocation at national and county levels  

### Success Metrics (Kenya Context)

| Metric        | Target Goal | Rationale                                                                 |
|---------------|------------|---------------------------------------------------------------------------|
| Recall (Sensitivity) | 65%        | Most important – catching two-thirds of high-risk patients is a major clinical improvement |
| Precision     | 40%        | Ensuring less than 60% of interventions are wasted on low-risk patients   |
| F1-Score      | 50%        | Balanced measure of model performance                                     |
| AUC           | 0.70       | Model discrimination ability                                              |
| Business Success | Clear, actionable policy recommendations | Model must translate to concrete interventions for Kenya's healthcare system |

### Stakeholders
- **Primary:** Ministry of Health Kenya, County Hospital Directors  
- **Secondary:** Healthcare Workers (Doctors, Nurses, Discharge Planners)  
- **Tertiary:** Diabetic Patients and Families, Health Insurance Providers  
- **Implementation Partners:** Kenya Medical Research Institute (KEMRI), World Health Organization (WHO) Kenya  

### Data Scope
- **Primary Dataset:** 101,766 patient encounters from 130 US hospitals (1999–2008)  
- **Key Variables:** 50 features including demographics, clinical data, medications, diagnoses  
- **Target Variable:** Readmission within 30 days (binary classification)  
- **Context Adaptation:** Model calibrated and recommendations tailored for Kenya's specific healthcare context  

### Project Constraints
- **Resource Limitations:** Must work within Kenya's resource-constrained environment  
- **Interpretability:** Clear explanations for predictions to gain healthcare worker trust  
- **Simplicity:** Implementable with limited technical infrastructure  
- **Cultural Relevance:** Interventions must be culturally appropriate and feasible  
- **Scalability:** Applicable across different hospital types and sizes in Kenya  

### Ethical Considerations
- **Patient Privacy:** No patient identifiers in model training or deployment  
- **Equity:** Ensure model does not disadvantage any demographic groups  
- **Transparency:** Clear communication about model limitations and uncertainties  
- **Clinical Oversight:** Model supports, does not replace, clinical judgment  
- **Consent:** Proper procedures for any data collection in deployment  

### Expected Deliverables
- **Predictive Model:** Random Forest classifier achieving 69% recall (exceeds target)  
- **Policy Brief:** 5-point actionable recommendations for Ministry of Health  
- **Demonstration Tool:** Streamlit application for healthcare worker training  
- **Technical Documentation:** Model specifications, validation results, maintenance guidelines  

### Project Justification
This project addresses **Sustainable Development Goal 3 (Good Health and Well-being)** and aligns with Kenya's **Universal Health Coverage** agenda. By applying machine learning to optimize resource allocation, it aims to:

- Improve healthcare outcomes for Kenya's estimated **500,000 diabetic patients**  
- Make more efficient use of limited public health resources


## Phase 2: Data Understanding

## Initial Setup and Data Loading

We began by setting up the working environment and loading the datasets. This step ensured that all required tools were available and allowed us to gain an initial understanding of the data structure.

### Key Setup and Loading Tasks

#### Importing Required Libraries
All necessary libraries were imported to support data manipulation, analysis, and visualization throughout the project.

#### Loading the Datasets
Both CSV files were loaded into the environment to make the data available for exploration and processing.

#### Checking Dataset Shapes
We examined the shape of each dataset (rows × columns) to understand their size and dimensionality.

#### Previewing the Data
The first few rows of each dataset were displayed to gain insight into the data structure, column names, and sample values.

#### Reviewing Data Types
Basic information about each dataset was reviewed to understand the data types of the features and identify any potential inconsistencies.


## Initial Data Exploration

Before performing any data cleaning, we carried out an initial exploration of the dataset to better understand its structure and quality. This step helped identify potential issues that could affect modeling.

### Key Exploration Tasks

#### Checking for Missing Values
We examined the dataset for missing values, which are represented by the symbol `?`. Identifying these values early helps determine appropriate handling methods such as removal or imputation.

#### Target Variable Distribution
We analyzed the distribution of the target variable to understand class balance. This is important for selecting suitable modeling techniques and evaluation metrics.

#### Data Types and Unique Values
We reviewed the data types of each feature (numerical or categorical) and inspected the number of unique values per column. This helped identify categorical variables, detect constant or low-variance features, and assess the need for encoding.


## Creating Binary Target Variable

Based on the project objective, we transformed the original `readmitted` column into a binary target variable to simplify the prediction task and align it with the problem definition.

### Target Variable Transformation

- A value of **1** represents patients who were readmitted within 30 days (`'<30'`).
- A value of **0** represents patients who were not readmitted within 30 days (`'NO'` or `'>30'`).

To ensure traceability, a copy of the original `readmitted` column was retained for reference. A new binary column, `readmitted_30_day`, was then created based on the defined mapping.

### Verification and Class Distribution

The transformation was verified to ensure correctness, and the distribution of the new binary target variable was examined. The results revealed a significant class imbalance, with approximately:

- **11.2%** of patients readmitted within 30 days  
- **88.8%** of patients not readmitted within 30 days  

This imbalance will need to be considered during model selection and evaluation.

### Important Observations from Step 2

- **Missing Values**: The `weight` column contains approximately **96.86% missing values**, making it a strong candidate for removal.
- **Target Variable Imbalance**: Only **11.16%** of patients were readmitted within 30 days, indicating a highly imbalanced classification problem.
- **ID Mapping Issue**: The ID mapping file appears to combine multiple mappings and will require separation into distinct reference tables before further use.

## Understanding and Separating ID Mappings

The ID mapping file provided with the dataset contains multiple code-to-description mappings combined into a single file. To ensure proper interpretation of categorical variables, we first needed to understand and separate these mappings.

### Key Mapping Tasks

#### Examining the Mapping File Structure
We reviewed the structure of the combined mapping file to understand how different ID types and their corresponding descriptions were organized.

#### Identifying Mapping Sections
Distinct sections within the file were identified, each representing a different type of ID mapping (e.g., admission type, discharge disposition, admission source).

#### Separating Individual Mappings
The combined file was separated into individual mapping DataFrames, with each DataFrame corresponding to a specific ID type. This separation allows for clearer and more accurate mapping.

#### Mapping IDs to the Main Dataset
The separated mappings were then used to map coded values in the main dataset to their descriptive labels.

#### Verification of Mappings
We verified that all ID values present in the main dataset had corresponding entries in the mapping tables to ensure data completeness and prevent unmapped or missing categories.

## Applying ID Mappings to the Main Dataset

After separating the ID mappings, we applied them to the main dataset to replace coded values with meaningful descriptive labels. This improves data interpretability while retaining the original IDs for reference.

### Mapping and Validation Tasks

#### Creating Descriptive Columns
The ID mappings were applied to create new descriptive columns while keeping the original ID columns intact.


##  Feature Engineering

To enhance model performance and improve interpretability, we engineered new features from existing variables. These transformations help capture meaningful patterns while reducing complexity.

### Feature Engineering Tasks

#### Simplifying Diagnosis Codes
Diagnosis codes were grouped from over **700 individual ICD codes** into approximately **20 broader diagnostic categories**. This reduces dimensionality while preserving clinical relevance.

#### Creating Age Groups
The existing age brackets were consolidated into broader age groups to better capture age-related trends and reduce sparsity.

#### Medication-Based Features
Several features were derived from medication-related columns to summarize treatment patterns:

- **Medication Change Count**: Number of medications that were changed or increased.
- **Steady Medication Count**: Number of medications maintained at a steady dosage.

#### Hospital Utilization Feature
A new feature representing **total hospital visits** was created by combining inpatient, outpatient, and emergency visit counts.

#### Diabetes Severity Indicator
A diabetes severity indicator was created based on relevant test results and clinical measurements, providing a summarized measure of disease severity.

### Outcome of Feature Engineering
These engineered features reduce noise, improve clinical interpretability, and provide richer inputs for downstream modeling.

## Exploratory Data Analysis (EDA) and Visualizations

Exploratory Data Analysis was conducted to uncover patterns, relationships, and trends in the data, with a particular focus on factors influencing patient readmission.

#### Categorical Variables vs Readmission
Bar charts were created to show readmission rates across key categorical variables, highlighting differences in readmission likelihood between categories.

#### Numerical Variables Analysis
Histograms and box plots were used to examine the distributions of numerical variables and compare them by readmission status.

#### Correlation Analysis
A correlation matrix of numerical features was generated to identify relationships between variables and detect potential multicollinearity.

#### Diagnosis Group Heatmap
A heatmap was created to visualize readmission rates across different diagnosis groups, helping identify high-risk medical categories.

#### Length of Hospital Stay Analysis
Additional analysis was performed to explore the relationship between readmission rates and the length of hospital stay.

#### Class Imbalance Check
The distribution of the target variable was revisited to confirm the extent of class imbalance before modeling.

### Summary of Key Findings
Key insights from the visual analysis were summarized to guide feature selection, modeling decisions, and interpretation of results.

Heatmap: Readmission Rate by Diagnosis Groups


<img width="929" height="712" alt="output_65_1" src="https://github.com/user-attachments/assets/27bbfda3-d4c4-4854-98e4-2543b9930415" />

    
Readmission Rate by Time in Hospital


<img width="712" height="424" alt="output_66_1" src="https://github.com/user-attachments/assets/96534855-1304-42d8-afe9-ba79c4a4d16e" />

     ======================================================================
    KEY INSIGHTS SUMMARY
    ======================================================================
    
    1. CLASS IMBALANCE:
       • Overall 30-day readmission rate: 11.16%
       • Not readmitted (<30 days): 90,409 patients (88.8%)
       • Readmitted (<30 days): 11,357 patients (11.2%)
       • Imbalance ratio: 8.0:1
    
    2. TOP RISK FACTORS (based on correlation):
       Features that INCREASE readmission risk (top 5):
    
       Features that DECREASE readmission risk (top 5):
    
    3. HIGH-RISK PATIENT GROUPS:
       • Age: Young Adult have highest rate: 11.8%
       • Top high-risk discharge types:
         1. return for outpatient ser: 66.7%
         2. Discharged: 44.4%
         3. this hospital: 42.9%
    
    4. CLINICAL INSIGHTS:
       • Time in hospital: Longer stays associated with higher readmission
       • Number of medications changed: Higher changes = higher risk
       • Emergency visits: More emergency visits = higher risk
       • Diagnosis: Certain diagnosis groups have much higher readmission rates

## Data Preparation for Modeling

In this step, the dataset was prepared for machine learning by selecting relevant features, encoding categorical variables, and splitting the data into training and testing sets. These steps ensure the data is ready for modeling while preserving its structure and distribution.

### Key Preparation Tasks

#### Feature Selection
Minimum Viable Product (MVP) features were selected based on project requirements and insights gathered from EDA and feature engineering.

#### Encoding Categorical Variables
Categorical features were encoded to make them compatible with machine learning models:

- **Label Encoding**: Used for simple models such as logistic regression.
- **One-Hot Encoding**: Used for advanced models like Random Forest and XGBoost to avoid ordinal assumptions.

#### Train-Test Split
The dataset was split into:

- **Training Set**: 80% of the data
- **Testing Set**: 20% of the data  

Stratification was applied to preserve the class distribution of the target variable across both sets.

#### Handling Class Imbalance
Preparations were made for addressing class imbalance in later modeling steps, ensuring fair evaluation of minority classes.

#### Saving Prepared Data
The training and testing sets were saved for reproducibility and later use in modeling.

#### Feature Importance Preview
A correlation-based overview of feature importance was generated to provide insights into influential variables.

#### Summary of Prepared Data
A concise summary of the prepared datasets, including feature counts, target distribution, and encoded columns, was reviewed to confirm readiness for modeling.


# Improving the Baseline Model
##  Improving Logistic Regression Model (Focused Approach)

To enhance the baseline Logistic Regression model, a systematic approach was applied focusing on feature improvement, class imbalance handling, and parameter optimization.

### Model Improvement Strategies

#### Better Feature Engineering
- Created more meaningful features using **medical domain knowledge**.
- Improved representation of risk factors and hospital utilization metrics.

#### Feature Selection
- Applied **statistical tests** to select the most relevant features.
- Removed redundant or low-impact variables to reduce noise and improve model performance.

#### Hyperparameter Tuning
- Conducted a **systematic search** for the best model parameters.
- Simplified the grid search for faster execution.
- Added **feature existence checks** to prevent KeyErrors.
- Improved error handling throughout the pipeline.

#### SMOTE Integration
- Implemented **SMOTE (Synthetic Minority Oversampling Technique)** within the pipeline to handle class imbalance effectively.
- Ensured minority class (patients readmitted within 30 days) is properly represented during training.

#### Threshold Optimization
- Determined the **optimal decision threshold** to achieve the target **Recall**, crucial for the Kenyan healthcare context.

### Comprehensive Evaluation
- Model performance was compared against the baseline Logistic Regression.
- Key metrics including **Recall, Precision, F1-Score, and Accuracy** were analyzed.
- Coefficients and feature importance were interpreted to provide **business insights**.
- The improved model demonstrates better identification of high-risk patients and informs actionable interventions in Kenyan hospitals.

#We'll systematically improve Logistic Regression through:
#1. Feature Engineering & Selection
#2. Hyperparameter Tuning"
#3. Handling Class Imbalance (SMOTE
#4. Threshold Optimization
#5. Cross-Validation


# FINAL RECOMMENDATIONS FOR IMPROVED LOGISTIC REGRESSION

**1 Model Performance Summary**
- **Recall target achieved:** 0.651 *(Target: 0.65)*  
- The model would identify approximately **1,479 high-risk patients**  
- **Precision target not met:** 0.131 *(Target: 0.40)*  

**2. Key Risk Factors Identified**
- **Emergency Ratio:** Increases readmission risk by **2.3× per unit**
- **Number of Medications Changed:** Decreases risk by **0.7× per unit**
- **Number of Emergency Visits:** Decreases risk by **0.8× per unit**

**3. Recommended Action**
- The model **meets the critical Recall requirement** for the Kenyan healthcare context
- Suitable for **deployment in identifying high-risk patients**
- Recommended **prediction threshold:** **0.46**

**4. Kenya Healthcare Context Implications**
- The model can identify **sufficient high-risk patients** for targeted interventions
- Enables **effective allocation of limited healthcare resources**
- Supports **proactive care** for patients at high risk of readmission


##  Visualizations for Improved Logistic Regression Model

To better understand the performance and practical implications of the improved Logistic Regression model, several visualizations were created.

### Key Visualizations

#### Confusion Matrix
- Illustrates the trade-off between **catching high-risk patients (Recall)** and generating **false alarms**.
- Helps assess how well the model identifies patients at risk of readmission.

#### ROC Curve
- Displays the model's **discrimination ability** between readmitted and non-readmitted patients.
- Area Under the Curve (AUC) provides a single measure of model performance.

#### Feature Importance
- Visualizes which features most strongly influence **readmission risk**.
- Helps healthcare practitioners focus on the most relevant risk factors.

#### Performance vs Targets
- Compares model metrics with **Kenya-specific targets** for Recall and Precision.
- Highlights which goals are achieved and areas needing improvement.

#### Precision-Recall Tradeoff
- Shows the relationship between **Recall and Precision** at different thresholds.
- Guides threshold selection to balance catching high-risk patients and minimizing false positives.

#### Business Impact Visualization
- Demonstrates the **practical implications** of the model in the healthcare context.
- Illustrates potential improvements in patient outcomes and resource allocation when high-risk patients are accurately identified.

Creating comprehensive visualizations for the improved model...



<img width="1144" height="1031" alt="output_105_1" src="https://github.com/user-attachments/assets/e03958e3-e984-43ef-8c01-c05687edb3ad" />

Creating final summary visualization...

<img width="718" height="424" alt="output_107_1" src="https://github.com/user-attachments/assets/5df7565a-8342-4341-8372-d106397e117d" />
    
## Key Insights from Visualizations
1. **Recall target achieved:** 65.1% *(meets Kenya healthcare requirement)*
2. **Precision remains low:** 13.1% *(indicates many false alarms)*
3. **Top risk factor identified:** Emergency Ratio *(increases readmission risk by 2.3×)*
4. **Business impact:** The model would identify approximately **1,479 high-risk patients**
5. **Performance trade-off:** Accepting a higher number of false positives in order to capture most high-risk patients


## Model 2: Random Forest Model

To improve predictive performance beyond Logistic Regression, a **Random Forest** model was built, tuned, and evaluated for readmission prediction.

### Model Training and Optimization

#### Initial Training
- Trained a Random Forest model with **balanced class weights** to address target class imbalance.

#### Hyperparameter Tuning
- Performed **Randomized Search** to optimize model parameters for better performance.
- Ensured efficient computation while exploring a wide range of parameter combinations.

#### Threshold Optimization
- Adjusted the decision threshold to **maximize Recall**, critical for identifying high-risk patients in the Kenyan healthcare context.

### Evaluation and Analysis

#### Feature Importance
- Conducted an **in-depth analysis** of feature importance.
- Identified the most influential predictors of readmission, providing actionable clinical insights.

#### Comparison with Logistic Regression
- Evaluated whether Random Forest achieved:
  - Higher **Recall** than Logistic Regression.
  - Improved **Precision** while maintaining Recall.
- Compared overall business impact, including the number of high-risk patients correctly identified.

#### Deployment Recommendations
- Discussed practical recommendations for implementing the model in Kenyan hospitals.
- Highlighted potential improvements in patient outcomes and resource allocation based on model predictions.

### Key Evaluation Questions

- Does Random Forest achieve **better Recall** than Logistic Regression?
- Does it **improve Precision** while maintaining Recall?
- What are the **top risk factors** according to the model?
- Is the **business impact** better than Logistic Regression?

7. Comparison with Logistic Regression Model
       --------------------------------------------------
    
       Model Comparison:
       ----------------------------------------------------------------------
       Metric           |  Logistic Reg  |  Random Forest  |  Target  |  Best Model
       ----------------------------------------------------------------------
       Recall          |   0.651✓     |   0.690✓       |   0.65   |   RF
       Precision       |   0.131✗     |   0.154✗       |   0.40   |   RF
       F1-Score        |   0.217✗     |   0.252✗       |   0.50   |   RF
       AUC             |   0.588✗     |   0.660✗       |   0.70   |   RF


### Random Forest Model Impact
- Identifies **1,567 out of 2,271 high-risk patients**  
  *(Recall: 69.0%)*
- Generates **8,615 false alarms**
- Achieves **Precision of 15.4%**  
  *(15.4% of flagged patients are actually high-risk)*

### Comparison with Logistic Regression
- **Additional high-risk patients identified:** 88
- **Change in false alarms:** −1,236
- **Precision improvement:** +0.023


# RANDOM FOREST MODEL SUMMARY

**1. Performance Summary**
- **Recall:** 0.690  *Meets Kenya target*
- **Precision:** 0.154  *Below target*
- **F1-Score:** 0.252  *Below target*
- **AUC:** 0.660  *Below target*

**2. Key Improvements Over Logistic Regression**
- **Recall improvement:** +0.039
- **Precision improvement:** +0.023
- **F1-Score improvement:** +0.034
- **AUC improvement:** +0.073

 **3. Top Risk Factors Identified**
- **Total Hospital Visits** — Importance: 0.4798  
- **Discharge Disposition 1** — Importance: 0.1515  
- **Discharge Disposition 16** — Importance: 0.1357  
- **Discharge Disposition 7** — Importance: 0.0784  
- **Number of Emergency Visits** — Importance: 0.0349  

**4. Recommendation for Kenya Healthcare Context**
- **Strength:** Critical Recall target achieved
- **Limitation:** Precision remains below target (many false alarms)
- **Conclusion:** Acceptable for the Kenyan healthcare context where **catching high-risk patients is the top priority**

**5. Next Steps**
- Random Forest meets the **critical Recall requirement**
- Evaluate the **trade-off between improved Precision and resource constraints**
- Select the final model based on **healthcare resource availability**


## Visualizations and Final Comparison

To provide a comprehensive understanding of model performance and practical implications, visualizations and comparisons were conducted across all models.

### Key Components

#### Visual Comparison of Performance Metrics
- Side-by-side visualizations of **Recall, Precision, F1-Score, and Accuracy** for all models.
- Highlights which models perform best under Kenyan healthcare priorities.

#### Business Impact Analysis
- Assesses how each model would affect patient outcomes and hospital resource allocation.
- Focuses on the number of **high-risk patients correctly identified**.

#### Feature Importance Interpretation
- Compares influential features across models.
- Provides actionable insights for **policy-making and clinical decision support**.

#### Clear Recommendations
- Identifies the **best-performing model** for deployment in Kenya.
- Provides justification based on performance metrics, business impact, and practicality.

#### Deployment Strategy
- Offers a step-by-step approach for integrating the chosen model into the **Kenyan healthcare system**.
- Suggests ways to monitor and update the model to maintain effectiveness over time.


Creating comprehensive visualizations and analysis...
    
    1. Model Performance Comparison

<img width="1288" height="884" alt="output_119_1" src="https://github.com/user-attachments/assets/239c5267-2904-4bb5-9f29-571c1731566d" />

     2. Detailed Model Comparison
       --------------------------------------------------
    
                   Metric Logistic Regression Random Forest    Target Improvement
                   Recall               0.651         0.690     0.650      +0.039
                Precision               0.131         0.154     0.400      +0.023
                 F1-Score               0.217         0.252     0.500      +0.034
                      AUC               0.588         0.660     0.700      +0.073
     High-Risk Identified               1,479         1,567  Maximize         +88
         High-Risk Missed                 792           704  Minimize         -88
             False Alarms               9,851         8,615  Minimize      -1,236
           Threshold Used                0.46          0.48   Optimal       +0.02


# Kenya Healthcare Context Analysis Based on the Two Model so far 


**Logistic Regression for Kenya**
- **Recall:** 65.1% *(Meets 65% target)*
- **Precision:** 13.1% *(Below 40% target)*
- Identifies approximately **1,479 high-risk patients**
- Generates **9,851 false alarms**
- **Resource implication:** For every 100 interventions, only **13** are for actual high-risk patients



**Random Forest for Kenya**
- **Recall:** 69.0% *(Meets 65% target)*
- **Precision:** 15.4% *(Below 40% target)*
- Identifies approximately **1,567 high-risk patients**  
  *(+88 more than Logistic Regression)*
- Generates **8,615 false alarms**  
  *(-1,236 fewer than Logistic Regression)*
- **Resource implication:** For every 100 interventions, about **15** are for actual high-risk patients


**Critical Considerations for Kenya (Limited Resources)**
- Both models capture **sufficient high-risk patients** *(Recall > 65%)*
- **Random Forest captures more high-risk patients** *(69.0% vs 65.1%)*
- **Random Forest produces fewer false alarms** *(8,615 vs 9,851)*
- **Random Forest is more resource-efficient** for the Kenyan healthcare system


## 4. Feature Importance for Policy Recommendations
---

### Common Important Features (Both Models)

#### 1. Hospital Utilization Patterns
- **Total hospital visits** *(most important feature in Random Forest)*
- **Emergency visits**
- **Time spent in hospital**

#### 2. Discharge Disposition
- Specific discharge types are **strong predictors** of readmission
- Outcomes differ significantly between **home discharge** and **other facilities**

---

### Unique Insights from Random Forest
1. **Discharge disposition is highly influential**, accounting for **40.3% of total feature importance**
2. **Specific discharge codes** *(1, 16, 7)* are critical predictors of readmission
3. **Hospital stay–related factors** account for **50.7% of total importance**

---

### Policy Implications for Kenya
1. **Target** patients with **multiple hospital visits**
2. **Focus** on improved **discharge planning and post-discharge follow-up**
3. **Monitor** emergency department utilization closely
4. **Intervene** before discharge for patients with **high-risk discharge dispositions**

---

## 5. Final Model Recommendation for Kenya
---

### Recommendation: **Deploy Random Forest Model**

---

### Why Random Forest?
-  **Higher Recall:** 69.0% vs 65.1% *(captures more high-risk patients)*
-  **Fewer false alarms:** 8,615 vs 9,851 *(better resource utilization)*
-  **Better discrimination:** AUC 0.660 vs 0.588
-  **Clearer risk factor identification** for policymaking

---

### Practical Considerations for Kenya
1. **Model Complexity:** More complex than Logistic Regression but still interpretable
2. **Computational Requirements:** Slightly higher but manageable
3. **Implementation:** Can be deployed as a **simple scoring or risk stratification system**
4. **Maintenance:** Feature importance supports **explainability and trust**

---

### Deployment Strategy
- **Phase 1:** Pilot the Random Forest model in **1–2 hospitals**
- **Phase 2:** Compare outcomes against **current clinical practice**
- **Phase 3:** Scale deployment to **regional hospitals** if successful
- **Phase 4:** **National rollout** with continuous performance monitoring


## Model 3:  XGBoost Model

To further improve predictive performance, an **XGBoost model** was developed, focusing on handling class imbalance and optimizing for high Recall in the Kenyan healthcare context.

### Model Training and Optimization

#### Handling Class Imbalance
- Applied **`scale_pos_weight`** to balance the minority class (patients readmitted within 30 days) during training.

#### Hyperparameter Tuning
- Conducted a **comprehensive hyperparameter search** to identify the best combination of parameters for optimal performance.
- Focused on boosting depth, learning rate, and number of estimators to improve predictive accuracy.

#### Threshold Optimization
- Determined the **decision threshold** that maximizes Recall while maintaining acceptable Precision.
- Ensures the model effectively identifies high-risk patients.

### Evaluation and Analysis

#### Feature Importance
- Performed a detailed analysis of feature importance to understand which factors most strongly influence readmission risk.
- Insights can guide hospital policies and targeted interventions.

#### Comparison with Other Models
- Compared XGBoost results with:
  - Baseline Logistic Regression
  - Improved Logistic Regression
  - Random Forest
- Assessed which model achieves the **best balance of Recall and Precision**.

#### Weighted Scoring Based on Kenya’s Priorities
- Performance metrics were combined using a **weighted scoring system**:
  - Recall contributes **50%** of the score to prioritize identifying high-risk patients.
  - Other metrics (Precision, F1-Score, Accuracy) contribute the remaining 50%.

#### Practical Deployment Considerations
- Discussed strategies for implementing the XGBoost model in **Kenyan hospitals**.
- Recommendations include monitoring model performance, updating features, and integrating predictions into clinical workflows for actionable interventions.



    
    7. Comprehensive Model Comparison
       --------------------------------------------------
    
       Performance Comparison (All Models):
       -------------------------------------------------------------------------------------
       Metric           |  Logistic Reg  |  Random Forest  |  XGBoost       |  Target  |  Best Model
       -------------------------------------------------------------------------------------
       Recall          |   0.651✓     |   0.690✓       |   0.655✓     |   0.65   |   RF
       Precision       |   0.131✗     |   0.154✗       |   0.163✗     |   0.40   |   XGB
       F1-Score        |   0.217✗     |   0.252✗       |   0.261✗     |   0.50   |   XGB
       AUC             |   0.588✗     |   0.660✗       |   0.672✗     |   0.70   |   XGB



##  XGBoost Performance Summary
- **Recall:** 0.655 *(Meets Kenya target)*
- **Precision:** 0.163 *(Below target)*
- **F1-Score:** 0.261 *(Below target)*
- **AUC:** 0.672 *(Below target)*



##  Comparison with Other Models
- **Compared to Logistic Regression:**  
  - Recall: +0.004  
  - Precision: +0.032  

- **Compared to Random Forest:**  
  - Recall: −0.035  
  - Precision: +0.009  



## Key Features Identified by XGBoost
- **Total Hospital Visits** — Importance: 0.3178  
- **Discharge Disposition** — Importance: 0.1899  
- **Number of Emergency Visits** — Importance: 0.0641  
- **Time in Hospital** — Importance: 0.0625  
- **Age (Numeric)** — Importance: 0.0585  



## COMPREHENSIVE THREE-MODEL COMPARISON VISUALIZATION

  Detailed Performance Comparison Table
 ------------------------------------------------------------
    
       Performance Metrics (Test Set):
       ------------------------------------------------------------------------------------------
                   Model Recall Precision F1-Score    AUC High-Risk ID    Missed False Alarms Threshold
     Logistic Regression  0.651     0.131    0.217  0.588        1,479       792        9,851      0.46
           Random Forest  0.690     0.154    0.252  0.660        1,567       704        8,615      0.48
                 XGBoost  0.655     0.163    0.261  0.672        1,487       784        7,645      0.48
                  TARGET  0.650     0.400    0.500  0.700     Maximize  Minimize     Minimize   Optimal
    

## 8. Business Impact Comparison Between The 3 Model


**High-Risk Patients Identified**
- **Logistic Regression:** 1,479 patients *(65.1% recall)*
- **Random Forest:** 1,567 patients *(69.0% recall)*
- **XGBoost:** 1,487 patients *(65.5% recall)*



**False Alarms (Resource Waste)**
- **Logistic Regression:** 9,851 false alarms *(Precision: 13.1%)*
- **Random Forest:** 8,615 false alarms *(Precision: 15.4%)*
- **XGBoost:** 7,645 false alarms *(Precision: 16.3%)*



### Patients Missed (Critical Errors)
- **Logistic Regression:** 792 missed  
  *(34.9% of high-risk patients)*
- **Random Forest:** 704 missed  
  *(31.0% of high-risk patients)*
- **XGBoost:** 784 missed  
  *(34.5% of high-risk patients)*


##  Model Selection Recommendation for Kenya


**Weighted Model Scores**  
*(Recall 50%, Precision 25%, AUC 15%, F1-Score 10%)*

- **Logistic Regression:** 0.577  
- **Random Forest:** **0.633**  
- **XGBoost:** 0.625  



## Recommended Model for Kenya: **Random Forest**


**Why Random Forest is Recommended**
- **Highest weighted performance score:** 0.633
- Provides a **good balance between predictive performance and interpretability**
- Produces **fewer false alarms** compared to Logistic Regression
- Offers **clear feature importance**, supporting evidence-based policy making


## Deployment Considerations for Kenya

### Random Forest Deployment in the Kenyan Healthcare System

#### Technical Requirements
1. **Computational resources:** Moderate  
2. **Model size:** Medium  
3. **Prediction speed:** Medium  
4. **Maintenance complexity:** Moderate  



#### Operational Requirements
1. **Staff training** on model interpretation and usage  
2. **Integration** with existing hospital information systems  
3. Establishment of a **monitoring and evaluation framework**  
4. **Regular model updates and validation** to maintain performance  



#### Resource Implications
1. **Initial implementation cost:** Medium  
2. **Ongoing maintenance:** Required  
3. **Training requirements:** Moderate  
4. **Scalability:** Good


#  FINAL RECOMMENDATIONS


Based on the comprehensive model comparison, **Random Forest** is recommended for deployment in Kenya as it provides the **best balance between high Recall, overall performance, and practical deployment considerations**.



## Next Steps for Kenya Implementation
1. **Finalize model selection:** Random Forest  
2. **Develop a deployment plan** in collaboration with the Ministry of Health  
3. **Derive policy recommendations** informed by model feature importance  
4. **Design and run a pilot implementation** in selected hospitals  
5. **Plan for scaling and nationwide deployment**


## Policy Recommendations and Deployment Planning


**Based on Random Forest Model Analysis for the Kenyan Healthcare System**  
**Model Performance:** Recall = **69.0%**, Precision = **15.4%**



## 1. Executive Summary for Ministry of Health

**Problem**
High rates of unplanned readmissions among diabetic patients place significant strain on Kenya’s limited healthcare resources and increase system-wide costs.

**Solution**
A **Random Forest machine learning model** that predicts 30-day readmission risk, achieving **69.0% Recall** in identifying high-risk patients.

**Impact**
- Identifies **1,567 out of 2,271 high-risk patients (69.0%)**
- Enables **targeted interventions** for patients most in need
- Optimizes allocation of scarce healthcare resources

**Investment**
- **Moderate implementation cost**
- **High potential return** through reduced readmissions and improved efficiency

---

## 2. Key Risk Factors for Kenyan Patients

**Top 5 Risk Factors (by Feature Importance)**

1. **Multiple Hospital Visits (Past Year)**
   - Feature importance: **0.480**
   - Implication: Patients with **≥3 hospital visits annually**

2. **Routine Home Discharge**
   - Feature importance: **0.151**
   - Implication: Standard discharge processes may need strengthening

3. **Discharge Disposition Code 16**
   - Feature importance: **0.136**
   - Implication: Requires further investigation into discharge pathway

4. **Discharge Disposition Code 7**
   - Feature importance: **0.078**
   - Implication: Another high-risk discharge category

5. **Frequent Emergency Department Visits**
   - Feature importance: **0.035**
   - Implication: Indicator of poor outpatient disease control

---

## 3. Policy Recommendations for Ministry of Health

 **1. Targeted Discharge Planning**
Implement enhanced discharge planning for high-risk patients.

**Recommended Actions:**
- Mandatory **30-minute discharge counseling**
- Provision of **home care kits**
- **Follow-up calls within 72 hours** of discharge

---

**2. Hospital Utilization Monitoring**
Actively manage frequent hospital visitors.

**Recommended Actions:**
- Flag patients with **≥3 hospital visits within 6 months**
- Assign **case managers**
- Develop **community-based care alternatives**

---

**3. Emergency Department Optimization**
Reduce unnecessary ED visits through improved primary care.

**Recommended Actions:**
- Strengthen **primary diabetes care**
- Establish **community diabetes clinics**
- Train **community health workers** in diabetes management

---

**4. Data-Driven Resource Allocation**
Use predictive insights to optimize healthcare resources.

**Recommended Actions:**
- Prioritize specialists and equipment for high-risk patients
- Allocate follow-up nurses based on predicted risk
- Use predictions for **bed management and staffing**

---

 **5. Patient Education and Empowerment**
Improve patient self-management capacity.

**Recommended Actions:**
- Develop culturally appropriate education materials
- Train patients on medication adherence and symptom monitoring
- Establish **diabetes patient support groups**

---

## 4. Implementation Roadmap (12 Months)

**Months 1–2: Preparation**
- Form implementation committee (MOH, hospitals, stakeholders)
- Secure funding and approvals
- Select **2–3 pilot hospitals**

**Months 3–4: Technical Setup**
- Deploy model in pilot hospitals
- Integrate with hospital information systems
- Train hospital staff

**Months 5–8: Pilot Implementation**
- Use model for discharge planning
- Implement targeted interventions
- Monitor outcomes and collect data

**Months 9–10: Evaluation**
- Analyze pilot results
- Gather feedback from staff and patients
- Refine model and workflows

**Months 11–12: Scale-Up Planning**
- Develop national scale-up plan
- Prepare training materials
- Secure funding for national rollout

---

## 5. Resource Requirements and Budget Estimate

**Human Resources**
- Project Manager (1 FTE)
- Data Analyst (1 FTE)
- IT Support (0.5 FTE)
- Clinical champions at each hospital

**Technical Infrastructure**
- Deployment server
- EMR system integration
- Security and backup systems
- Training manuals and documentation

**Training and Capacity Building**
- Staff training sessions
- Training-of-trainers program
- Ongoing technical support

**Monitoring and Evaluation**
- Data collection systems
- Performance dashboards
- Regular evaluation reports

**Estimated Pilot Budget (2–3 Hospitals)**
| Item | Cost (KES) |
|---|---|
| Personnel | 2,500,000 |
| Technical Infrastructure | 1,200,000 |
| Training | 800,000 |
| Monitoring & Evaluation | 500,000 |
| Contingency (15%) | 750,000 |
| **Total** | **5,750,000** |

---

## 6. Expected Outcomes and Impact

**Clinical Outcomes**
- 20–30% reduction in 30-day readmissions
- Improved patient satisfaction
- Better management of diabetes complications

**Operational Outcomes**
- Improved bed utilization
- Reduced ED congestion
- Better use of specialist time

**Financial Outcomes**
- Cost savings from fewer readmissions
- Improved efficiency of healthcare spending

**System Outcomes**
- Stronger data-driven decision-making
- Improved care coordination
- Foundation for predictive analytics expansion

---

## 7. Risk Mitigation Strategies

| Risk | Mitigation |
|---|---|
| System integration challenges | Phased rollout with technical support |
| Staff resistance | Early engagement, training, clinical champions |
| Data quality issues | Validation checks and regular audits |
| Sustainability | Integrate into routine operations |
| Ethical concerns | Clear communication and opt-out mechanisms |

---

## 8. Success Metrics and Monitoring

**Primary Metrics**
- 30-day readmission rate
- Percentage of high-risk patients identified
- Reduction in false alarms

**Process Metrics**
- Number of targeted interventions delivered
- Staff protocol compliance
- System uptime

**Patient Metrics**
- Patient satisfaction
- Medication adherence
- ED visit frequency

**Financial Metrics**
- Cost per readmission prevented
- Resource utilization efficiency
- Return on investment


=

# Final Call to Action

Kenya can significantly improve diabetic patient outcomes through **data-driven predictive analytics**.

### We recommend:
1. **Approve** pilot implementation in 2–3 hospitals  
2. **Allocate** KES **5.75M** for the pilot phase  
3. **Establish** a monitoring and evaluation framework  
4. **Plan** for national scale-up based on pilot results  

This initiative positions Kenya as a leader in healthcare innovation in Africa and improves the lives of thousands of diabetic patients.



## Appendices: Technical Details

**Model Specifications**
- Algorithm: Random Forest Classifier
- Features: 48 (after encoding)
- Training samples: 81,412
- Test samples: 20,354
- Optimal threshold: 0.48
- Recall: 69.0%
- Precision: 15.4%

**Deployment Requirements**
- Hardware: 16GB RAM, 4-core CPU server
- Software: Python 3.8+, scikit-learn, pandas, numpy
- Integration: REST API
- Security: Encryption and access controls

**Model Maintenance**
- Retraining: Every 6 months
- Monitoring: Monthly performance reports
- Version control with rollback support



**Prepared for:** Ministry of Health, Kenya  
**Date:** January 17, 2026  
**Project:** Optimizing Healthcare Resource Allocation for Diabetic Patients
**@MoringaCapstoneProject_Group4**


