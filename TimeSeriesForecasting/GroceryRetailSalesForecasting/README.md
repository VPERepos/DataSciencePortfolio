# Retail Demand forcasting
## Problem definition
This project is based on the dataset M5-Forecasting-accuracy (https://www.kaggle.com/competitions/m5-forecasting-accuracy). The goal is to predict demand for the next 28 days for around 3500 items in 10 stores from 3 states in USA. The main challenge here is that the signals are very sparse and stochastic for each item. 
## Technologies and models used
In order to solve this problem such a classical time series forcaster like **Prophet** was applied to cumulative sales forecasting. For final prediction of the demand forecast for each item a gradient boosted decision trees based model called **LightDBM** was chosen as a best performing model. The data were loaded and manipulated by **Pandas** and **NumPy** libraries.   