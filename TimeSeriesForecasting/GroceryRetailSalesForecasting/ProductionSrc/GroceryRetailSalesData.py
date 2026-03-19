import pandas as pd
import numpy as np
import os

DATASETS_PATH = os.getenv("DATASETS_PATH") 

class GroceryRetailSalesData:
    def __init__(self):
        self._df_calendar = pd.read_csv(DATASETS_PATH + "/RetailOptimizationProject/m5-forecasting-accuracy/calendar.csv")
        self._df_sell_prices = pd.read_csv(DATASETS_PATH + "/RetailOptimizationProject/m5-forecasting-accuracy/sell_prices.csv")
        self._df_sales_train = pd.read_csv(DATASETS_PATH + "/RetailOptimizationProject/m5-forecasting-accuracy/sales_train_validation.csv")
        self._df_sales_evaluate = pd.read_csv(DATASETS_PATH + "/RetailOptimizationProject/m5-forecasting-accuracy/sales_train_evaluation.csv")
        self._store_ids = self._df_sales_train['store_id'].unique()
