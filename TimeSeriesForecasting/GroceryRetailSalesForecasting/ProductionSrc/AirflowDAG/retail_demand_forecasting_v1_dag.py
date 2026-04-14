from airflow.sdk import dag, task
from datetime import datetime
import pandas as pd
import os
import psycopg2

import sys
sys.path.append('/opt/airflow/src')

from GroceryRetailSalesForecastingProductionFunctions import load_data_to_production_database

PRODUCTION_DATABASE_URL = os.getenv("PRODUCTION_DATABASE_URL")

@dag(
    dag_id="retail_demand_forecasting_v1",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
)
def retail_demand_forecasting():

    @task
    def load_files_to_database(execute_step):
        if(execute_step):
            print("Step 1: Load Data to Production Database.")
            load_data_to_production_database("/opt/airflow/data/", PRODUCTION_DATABASE_URL)
        
    @task
    def step2():
        print("Step 2: Feature engineering")

    @task
    def step3():
        print("Step 3: Train model")

    @task
    def step4():
        print("Step 4: Save predictions")

    s1 = load_files_to_database(False)
    s2 = step2()
    s3 = step3()
    s4 = step4()

    s1 >> s2 >> s3 >> s4
    
retail_demand_forecasting()