# A/B Testing of an advertisment campaign.
## Problem definition.
This project is based on the the following dataset from kaggle: https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing?resource=download . 

The goal is to analyze the effectiveness of a digital advertising campaign using A/B testing and statistical modeling.

The main objective is to determine whether the advertisement campaign (“ad” group) leads to a significantly higher conversion rate compared to the control group ("psa” group), where users receive public service announcements instead of promotional ads.

## Technologies and Models Used
A **two-proportion z-test** and **bootstrap test** were used in order to check the hypothesis about the impact of advertisement in control and treatment groups. **Logistic regression** was conducted for estimating how big the impact of advertisement is and how the conversion probability correlates with measured parameters such as volume of ads shown and time and day with most ads. The data were loaded and manipulated by **Pandas**, **NumPy** and **statsmodels** libraries.