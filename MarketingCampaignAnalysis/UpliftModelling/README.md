# Uplift modeling of an advertisment campaign.
## Problem definition.
This project is based on the Kevin Hillstrom's dataset for uplift analytics of an email marketing campaign.

The goal is to train an uplift model and calculate the uplift score for each customer. Considering this score the customers are divided into groups like 'pursuadables', 'sure thing' and 'do not disturb'. The profile of the pursuadables is then created considering the available features.

## Technologies and Models Used
An **ensamble uplift model using lightGBM** was found to be most successfull for this task, comparing to **tree based models from scikit-uplift**. The data were loaded and manipulated by **Pandas** and **NumPy**  libraries.