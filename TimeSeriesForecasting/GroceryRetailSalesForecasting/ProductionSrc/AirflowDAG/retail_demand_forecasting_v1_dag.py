from airflow.sdk import dag, task
from datetime import datetime
import pandas as pd
import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

@dag(
    dag_id="retail_demand_forecasting_v1",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
)
def retail_demand_forecasting():

    @task
    def load_files_to_database():
        file_path = "/opt/airflow/data/calendar.csv"
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS working_hours_long (
                worker_id BIGINT,
                week_label TEXT,
                minutes BIGINT,

                CONSTRAINT fk_worker
                    FOREIGN KEY (worker_id)
                    REFERENCES workers(worker_id)
            );
        """)
        

    @task
    def step2():
        print("Step 2: Feature engineering")

    @task
    def step3():
        print("Step 3: Train model")

    @task
    def step4():
        print("Step 4: Save predictions")

    s1 = load_files_to_database()
    s2 = step2()
    s3 = step3()
    s4 = step4()

    s1 >> s2 >> s3 >> s4

retail_demand_forecasting()