import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2.extras import execute_values

def load_data_to_production_database(data_path, database_url):
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS calendar (
            date DATE,
            wm_yr_wk BIGINT,
            weekday TEXT,
            wday SMALLINT,
            month SMALLINT,
            year SMALLINT,
            d TEXT  PRIMARY KEY,
            event_name_1 TEXT,
            event_type_1 TEXT,
            event_name_2 TEXT,
            event_type_2 TEXT,
            snap_CA SMALLINT,
            snap_TX SMALLINT,
            snap_WI SMALLINT
            
        );
    """)

    with open(data_path + "/calendar.csv", "r") as f:
        cur.copy_expert("""
            COPY calendar (date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI)
            FROM STDIN
            WITH (FORMAT CSV, HEADER TRUE)
        """, f)

    conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sell_prices (
            store_id TEXT,
            item_id TEXT,
            wm_yr_wk BIGINT,
            sell_price NUMERIC
                        
        );
    """)

    with open(data_path + "/sell_prices.csv", "r") as f:
        cur.copy_expert("""
            COPY sell_prices (store_id,item_id,wm_yr_wk,sell_price)
            FROM STDIN
            WITH (FORMAT CSV, HEADER TRUE)
        """, f)

    conn.commit()

    df_sales_wide = pd.read_csv(data_path + "/sales_train_evaluation.csv")

    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_cols = [c for c in df_sales_wide.columns if c.startswith("d_")]

    df_sales_long = (
        df_sales_wide.set_index(meta_cols)[value_cols]
        .stack()
        .reset_index()
    )

    df_sales_long.columns = meta_cols + ["day", "sales"]
    df_sales_long["sales"] = df_sales_long["sales"].fillna(0).astype("int32")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales_long (
            id TEXT,
            item_id TEXT,
            dept_id TEXT,
            cat_id TEXT,
            store_id TEXT,
            state_id TEXT,
            day TEXT,
            sales INTEGER
        );
        """)

    conn.commit()

    data = list(df_sales_long.itertuples(index=False, name=None))

    query = """
    INSERT INTO sales_long (
        id, item_id, dept_id, cat_id,
        store_id, state_id, day, sales
    ) VALUES %s
    """

    execute_values(cur, query, data, page_size=5000)

    conn.commit()

    cur.close()
    conn.close()