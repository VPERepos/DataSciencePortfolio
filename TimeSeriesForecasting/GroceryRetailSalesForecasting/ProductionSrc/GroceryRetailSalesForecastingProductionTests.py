import pandas as pd
import numpy as np
import os
import psycopg2

from GroceryRetailSalesForecastingProductionFunctions import load_data_to_production_database

DATASETS_PATH = os.getenv("DATASETS_PATH")
PRODUCTION_DATABASE_URL = os.getenv("PRODUCTION_DATABASE_URL")

# Test first step in the production DAG: loading of data into production database

load_data_to_production_database(DATASETS_PATH, PRODUCTION_DATABASE_URL)

