import pytest
import logging
from GroceryRetailSalesData import GroceryRetailSalesData, CumulativeSalesTimeSeriesData
from GroceryRetailSalesForecastingModellingFunctions import generate_cumulative_sales_time_series


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

initial_data = GroceryRetailSalesData()
cumulative_sales_time_series_data = CumulativeSalesTimeSeriesData()

def test_grocery_retail_data_class():

    assert(initial_data._df_calendar.columns[0] == 'date')
    assert(initial_data._df_calendar.columns[1] == 'wm_yr_wk')
    assert(initial_data._df_calendar.columns[2] == 'weekday')
    assert(initial_data._df_calendar.columns[3] == 'wday')
    assert(initial_data._df_calendar.columns[4] == 'month')
    assert(initial_data._df_calendar.columns[5] == 'year')
    assert(initial_data._df_calendar.columns[6] == 'd')
    assert(initial_data._df_calendar.columns[7] == 'event_name_1')
    assert(initial_data._df_calendar.columns[8] == 'event_type_1')
    assert(initial_data._df_calendar.columns[9] == 'event_name_2')
    assert(initial_data._df_calendar.columns[10] == 'event_type_2')
    assert(initial_data._df_calendar.columns[11] == 'snap_CA')
    assert(initial_data._df_calendar.columns[12] == 'snap_TX')
    assert(initial_data._df_calendar.columns[13] == 'snap_WI')
    assert(initial_data._df_calendar.shape[0] == 1969)

    assert(initial_data._df_sell_prices.columns[0] == 'store_id')
    assert(initial_data._df_sell_prices.columns[1] == 'item_id')
    assert(initial_data._df_sell_prices.columns[2] == 'wm_yr_wk')
    assert(initial_data._df_sell_prices.columns[3] == 'sell_price')
    assert(initial_data._df_sell_prices.shape[0] == 6841121)

    assert(initial_data._df_sales_validation.shape[0] == 30490)
    assert(initial_data._df_sales_validation.shape[1] == 1919)

    assert(initial_data._df_sales_evaluation.shape[0] == 30490)
    assert(initial_data._df_sales_evaluation.shape[1] == 1947)

    assert(initial_data._store_ids[0] == 'CA_1')
    assert(initial_data._store_ids[1] == 'CA_2')
    assert(initial_data._store_ids[2] == 'CA_3')
    assert(initial_data._store_ids[3] == 'CA_4')
    assert(initial_data._store_ids[4] == 'TX_1')
    assert(initial_data._store_ids[5] == 'TX_2')
    assert(initial_data._store_ids[6] == 'TX_3')
    assert(initial_data._store_ids[7] == 'WI_1')
    assert(initial_data._store_ids[8] == 'WI_2')
    assert(initial_data._store_ids[9] == 'WI_3')


def test_generate_cumulative_sales_time_series():
    cumulative_sales_time_series_data = generate_cumulative_sales_time_series(initial_data)
