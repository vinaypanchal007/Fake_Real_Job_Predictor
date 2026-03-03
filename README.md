#Real_-_Fake_Job_Predictor

##A hybrid machine learning project to detect fraudulent job postings using NLP, structured metadata, and Logistic Regression.

##This project demonstrates how text processing, structured feature engineering, class imbalance handling, model tuning, and web deployment work together in a practical fraud detection system.

##Dataset Used
The dataset is taken from Kaggle: Real or Fake Job Posting Prediction.

Labels:
0 → Real Job
1 → Fake Job

The dataset contains only binary labels.
There is no separate "Unsure" label in the original data.

The dataset is highly imbalanced (~95% real, ~5% fake), so special techniques are used to handle this imbalance.

#Model Architecture
This project uses a hybrid pipeline combining textual and structured features.

1. Text Features (NLP)
Title
Company profile
Description
Requirements
Benefits

These fields are combined into a single text column and transformed using:
TF-IDF (unigrams + bigrams)
Stopword removal
Maximum 5000 features

2. Structured Metadata Features

Categorical features:
Location
Department
Employment type
Required experience
Required education
Industry
Function

Numeric features:
Extracted minimum salary from salary range
Telecommuting (0/1)
Has company logo (0/1)
Has screening questions (0/1)

Categorical features are encoded using OneHotEncoder.
Numeric features are scaled using StandardScaler.

3. Class Imbalance Handling

Because the dataset is highly imbalanced:
SMOTE (Synthetic Minority Oversampling Technique) is applied on the training data only.
This significantly improves fraud recall without harming overall accuracy.

4. Model

Logistic Regression
Hyperparameter tuning using GridSearchCV
5-fold cross-validation
ROC-AUC evaluation
Model Performance (Hybrid Tuned Model)
Test Accuracy ≈ 0.98–0.99
Fraud Precision ≈ 0.85+
Fraud Recall ≈ 0.85+
ROC-AUC ≈ 0.98

The model maintains strong fraud detection while minimizing false positives.

##Decision Logic (Application Layer)
Although the dataset is binary (0 or 1), the web application introduces a third category:
Fake probability ≤ 0.30 → Real Job
Fake probability between 0.30 and 0.70 → Unsure
Fake probability ≥ 0.70 → Fake Job
The “Unsure” category is added to reduce overconfident borderline predictions and improve reliability.

##Web Application
A Streamlit application is built to:
Accept job text details
Accept structured job metadata
Include credibility signals (logo, telecommuting, screening questions)
Output classification as Real, Fake, or Unsure
The raw probability is hidden by default to prevent misuse or misinterpretation.

##Feature Insights
The model learned meaningful fraud indicators such as:
Fraud-associated signals:
Words like “earn”, “money”, “data entry”
Work-from-home patterns
Certain metadata distributions

##Real-job indicators:
Corporate terms like “team”, “clients”, “enterprise”
Presence of company logo
Screening questions
This confirms that both textual and structured features contribute to fraud detection.

##Limitations
The dataset contains only binary labels.
Dataset bias may influence location or industry signals.
Subtle, well-written scams may still evade detection.
Performance depends on how representative the training data is of real-world postings.

##Learning Outcomes
This project demonstrates:
NLP feature engineering using TF-IDF
Hybrid modeling (text + structured data)
Handling class imbalance with SMOTE
Hyperparameter tuning and cross-validation
Model interpretability via feature importance
Deployment using Streamlit
