# Customer Churn and Survival Analysis.
## Problem definition.
This project is based on the IBM Customer Churn dataset.

The goal is first to train a classifier, which predicts the probability of a customer to churn. Secondary, the importance of the features for customer churn has to be estimated by a survival model, the scores are then used to segment customers into risk groups and estimate their probabilities to stay over time. This enables informed bussiness decisions for customer churn prevention treatments. 

## Technologies and Models Used
For Churn Probability estimation as the baseline estimator a simple **Logistic Regression** was trained and then compared with **Random Forest**, **XGBoost** and **LightGBM** classifiers. For survival analysis an implementation of the **Cox survival model** from **Scikit-survival** was used. The function for generating **Kaplan-Meier Curves** was implemented manually for estimating probability over time. The data were loaded and manipulated by **Pandas** and **NumPy**  libraries.