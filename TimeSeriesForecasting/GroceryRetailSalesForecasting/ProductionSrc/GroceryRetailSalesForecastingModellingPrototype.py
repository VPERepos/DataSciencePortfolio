from GroceryRetailSalesData import GroceryRetailSalesData

from GroceryRetailSalesForecastingModellingFunctions import generate_cumulative_sales_time_series
from GroceryRetailSalesForecastingModellingFunctions import fit_prophet_models
from GroceryRetailSalesForecastingModellingFunctions import generate_prophet_forecasts
from GroceryRetailSalesForecastingModellingFunctions import generate_item_prices_per_day
from GroceryRetailSalesForecastingModellingFunctions import calculate_price_indexes_per_store
from GroceryRetailSalesForecastingModellingFunctions import calculate_initial_feature_table
from GroceryRetailSalesForecastingModellingFunctions import calculate_laggs_for_feature_table
from GroceryRetailSalesForecastingModellingFunctions import train_regressor
from GroceryRetailSalesForecastingModellingFunctions import load_xgb_models
from GroceryRetailSalesForecastingModellingFunctions import evaluate_forecast_of_cumulative_sales
from GroceryRetailSalesForecastingModellingFunctions import calculate_predictions

from WRMSSEEvaluator import WRMSSEEvaluator

from GroceryRetailSalesVisualizationUtils import plot_prophet_forecasts_validation
from GroceryRetailSalesVisualizationUtils import plot_price_indexes
from GroceryRetailSalesVisualizationUtils import plot_cumulative_sales_preds_vs_actual

import numpy as np

initial_data = GroceryRetailSalesData()

cumulative_sales_time_series_data = generate_cumulative_sales_time_series(initial_data)

prophet_models = fit_prophet_models(cumulative_sales_time_series_data)

prophet_forecasts = generate_prophet_forecasts(prophet_models)

#plot_prophet_forecasts_validation(prophet_models, prophet_forecasts, initial_data._store_ids)

item_prices_per_day = generate_item_prices_per_day(initial_data)

price_indexes_per_store = calculate_price_indexes_per_store(initial_data, item_prices_per_day)

#plot_price_indexes(price_indexes_per_store, initial_data._store_ids)

initial_feature_table = calculate_initial_feature_table(
    prophet_forecasts,
    price_indexes_per_store,
    cumulative_sales_time_series_data,
    initial_data
)

feature_table_with_laggs, feature_table_with_laggs_validation = calculate_laggs_for_feature_table(initial_feature_table)

#train_regressor(feature_table_with_laggs_validation)

loaded_models = load_xgb_models()

eval_df_by_stores = evaluate_forecast_of_cumulative_sales(loaded_models, feature_table_with_laggs, initial_data._store_ids)

#plot_cumulative_sales_preds_vs_actual(eval_df_by_stores)

print("Calculating predicitons")

preds_static_shares = calculate_predictions(eval_df_by_stores, initial_data)

# train_df = pd.read_csv("sales_train_evaluation.csv")
# calendar = pd.read_csv("calendar.csv")
# prices = pd.read_csv("sell_prices.csv")

print("Starting score evaluation")

evaluator = WRMSSEEvaluator(initial_data._df_sales_evaluation, initial_data._df_calendar, initial_data._df_sell_prices)

actuals = initial_data._df_sales_evaluation.iloc[:, -28:].values  # last 28 days
preds_random = np.random.rand(30490, 28)        # replace with your model
preds_zeros = np.zeros((30490, 28), dtype=int)

# Sanity check:
print("WRMSSE_actual_actual: ", evaluator.score(actuals, actuals))  # should be ~0

score_random = evaluator.score(preds_random, actuals)
print("WRMSSE_random:", score_random)

score_zeros = evaluator.score(preds_zeros, actuals)
print("WRMSSE_zeros:", score_zeros)

score_static_shares = evaluator.score(preds_static_shares, actuals)
print("WRMSSE_static_shares:", score_static_shares)

"""
WRMSSE_actual_actual:  0.0
WRMSSE_random: 3.7036320434881547
WRMSSE_zeros: 5.449542341759063
WRMSSE_static_shares: 2.204506741759602
"""





