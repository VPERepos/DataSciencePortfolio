from GroceryRetailSalesData import GroceryRetailSalesData
from prophet import Prophet
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt

def generate_cumulative_sales_time_series(initial_data: GroceryRetailSalesData):
    result = {}

    for store in initial_data._store_ids:
        df_store_evaluation = initial_data._df_sales_evaluation[initial_data._df_sales_evaluation['store_id'] == store].copy()
        day_cols_evaluation = [col for col in df_store_evaluation.columns if col.startswith('d_')]
        df_sum_evaluation = df_store_evaluation[day_cols_evaluation].sum(axis=0).reset_index()
        df_sum_evaluation.columns = ['d', 'y']
        df_sum_evaluation = df_sum_evaluation.merge(initial_data._df_calendar[['d','date']], on='d', how='left')
        result[store] = df_sum_evaluation.copy()

    return result

def fit_prophet_models(time_series_data):
    result_models = {}
   
    for store, df in time_series_data.items():
        df_loc = df[['date','y']].rename(columns={'date':'ds'})
        model = Prophet(interval_width=0.95) 
        model.fit(df_loc)
        result_models[store] = model
    
    return result_models

def generate_prophet_forecasts(prophet_models, future_days = 28):
    result_prophet_forecasts = {}
    
    for store_id, model in prophet_models.items():
        
        future = model.make_future_dataframe(periods=future_days)
        
        forecast = model.predict(future)
         
        forecast_store = forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].copy()
        
        forecast_store['store_id'] = store_id
        
        result_prophet_forecasts[store_id] = forecast_store.copy()
    
    return result_prophet_forecasts

def generate_item_prices_per_day(initial_data: GroceryRetailSalesData):
    result_store_prices = {}
    for store in initial_data._store_ids:
        df_store_prices = initial_data._df_sell_prices[initial_data._df_sell_prices['store_id'] == store].copy()
        store_sell_prices = df_store_prices.copy()

        pivot_prices = store_sell_prices.pivot_table(
        index='wm_yr_wk',
        columns='item_id',
        values='sell_price',
        aggfunc='first'
        )
        
        original_items = initial_data._df_sales_validation[initial_data._df_sales_validation['store_id'] == store]['item_id'].tolist()
        
        original_items = list(dict.fromkeys(original_items))
        
        pivot_prices = pivot_prices.reindex(columns=original_items)
        
        df_day_week = initial_data._df_calendar[['d','wm_yr_wk']]
        
        result_store_prices[store] = df_day_week.merge(
            pivot_prices.reset_index(),
            on='wm_yr_wk',
            how='left'
        ).set_index('d')
        
        result_store_prices[store] = result_store_prices[store].drop(columns='wm_yr_wk', errors='ignore')
        
        result_store_prices[store] = result_store_prices[store].fillna(0)
    
    return result_store_prices

def calculate_price_indexes_per_store(initial_data, item_prices_per_day):
    day_cols = [col for col in initial_data._df_sales_validation.columns if col.startswith('d_')]
    
    total_sales_per_item = (
        initial_data._df_sales_validation[day_cols]
        .sum(axis=1)
        .groupby(initial_data._df_sales_validation['item_id'])
        .sum()
    )

    item_cat_map = initial_data._df_sales_validation[['item_id', 'cat_id']].drop_duplicates()
    sales_with_cat = total_sales_per_item.reset_index(name='total_sales').merge(
        item_cat_map,
        on='item_id',
        how='left'
    )
    total_sales_per_category = (
        sales_with_cat
        .groupby('cat_id')['total_sales']
        .sum()
    )

    sales_with_cat['cat_total'] = sales_with_cat['cat_id'].map(total_sales_per_category)

    sales_with_cat['weight'] = (
        sales_with_cat['total_sales'] / sales_with_cat['cat_total']
    )
    weights_by_cat = {}

    for cat in ['FOODS', 'HOBBIES', 'HOUSEHOLD']:
        df_cat = sales_with_cat[sales_with_cat['cat_id'] == cat]
        
        weights_by_cat[cat] = df_cat.set_index('item_id')['weight']

    features_per_store = {}

    for store in initial_data._store_ids:
        price_df = item_prices_per_day[store]
        
        features = pd.DataFrame(index=price_df.index)
        
        for cat in ['FOODS', 'HOBBIES', 'HOUSEHOLD']:
            
            weights = weights_by_cat[cat]
            
            # Align columns
            common_items = price_df.columns.intersection(weights.index)
            
            features[f'{cat}_price_index'] = (
                price_df[common_items] @ weights.loc[common_items]
            )
        features_per_store[store]=features.copy()

    normalized_features_per_store = {}

    for key, df in features_per_store.items():
        normalized_features_per_store[key] = features_per_store[key] / features_per_store[key].max()
    
    return normalized_features_per_store

def calculate_initial_feature_table(
        prophet_forecasts,
        price_indexes_per_store,
        cumulative_sales_time_series_data,
        initial_data
):
    initial_features_table = {}

    calendar = initial_data._df_calendar.copy()
    calendar = calendar[[
        'date', 'd', 'wday', 'month',
        'event_name_1','event_type_1',
        'event_name_2','event_type_2',
        'snap_CA','snap_TX','snap_WI'
    ]]
    calendar['date'] = pd.to_datetime(calendar['date'])

    final_dfs = []

    for store in initial_data._store_ids:
       
        df_forecast = prophet_forecasts[store].copy()

        df_prices = price_indexes_per_store[store] #.reset_index()  # 'd' becomes column

        df_sales = cumulative_sales_time_series_data[store].copy()

        df = df_prices.merge(df_sales, on='d', how='left')
        
        df = df.merge(calendar, on='d', how='left')
        df["date"] = df["date_x"]
        df = df.drop(columns=["date_x", "date_y"])
        df['date'] = pd.to_datetime(df['date'])

        df = df.merge(
            df_forecast[['ds', 'yhat']], #df_forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']],
            left_on='date',
            right_on='ds',
            how='left'
        )
        df.drop(columns='ds', inplace=True)

        state = store.split('_')[0]
        df['state'] = state
        df['store_id'] = store

        snap_col = f'snap_{state}'
        df['snap'] = df[snap_col]

        """
        df = df[[
            'date','wday','month',
            'event_name_1','event_type_1',
            'event_name_2','event_type_2',
            'state','store_id','snap',
            'yhat_lower','yhat_upper','yhat',
            'FOODS_price_index','HOBBIES_price_index','HOUSEHOLD_price_index',
            'y'
        ]]
        """
        df = df[[
            'date','wday','month',
            'event_name_1','event_type_1',
            'event_name_2','event_type_2',
            'state','store_id','snap',
            'yhat',
            'FOODS_price_index','HOBBIES_price_index','HOUSEHOLD_price_index',
            'y'
        ]]
        df[['event_name_1','event_type_1','event_name_2','event_type_2']] = \
            df[['event_name_1','event_type_1','event_name_2','event_type_2']].fillna('None')
        
        initial_features_table[store]=df.copy()
    
    return initial_features_table

def calculate_laggs_for_feature_table(initial_features_table, future_days = 28):
    table_with_laggs = {}
    table_with_laggs_validation = {}
    keys = list(initial_features_table.keys())

    for store in keys:
        initial_table = initial_features_table[store]
        yhat_future_cols = {}
        yhat_past_cols = {}
        y_past_cols = {}
        y_future_cols = {}
        
        wday_future = {}
        month_future = {}
        event_name_1_future = {}
        event_type_1_future = {}
        event_name_2_future = {}
        event_type_2_future = {}
        
        for i in range(future_days):
            wday_future[f'wday_future_{i+1}'] = initial_table['wday'].shift(-(i + 1))
            month_future[f'month_future_{i+1}'] = initial_table['month'].shift(-(i + 1))
            event_name_1_future[f'event_name_1_future_{i+1}'] = initial_table['event_name_1'].shift(-(i + 1))
            event_type_1_future[f'event_type_1_future_{i+1}'] = initial_table['event_type_1'].shift(-(i + 1))
            event_name_2_future[f'event_name_2_future_{i+1}'] = initial_table['event_name_2'].shift(-(i + 1))
            event_type_2_future[f'event_type_2_future_{i+1}'] = initial_table['event_type_2'].shift(-(i + 1))
            yhat_future_cols[f'yhat_future_{i+1}'] = initial_table['yhat'].shift(-(i + 1))
            yhat_past_cols[f'yhat_past_-{i+1}'] = initial_table['yhat'].shift((i + 1))
            y_past_cols[f'y_past_-{i+1}'] = initial_table['y'].shift(i + 1)
            y_future_cols[f'y_future_{i+1}'] = initial_table['y'].shift(-(i + 1))
            
        
        # Concatenate all new columns at once
        future_and_past_shifts = pd.concat([
            pd.DataFrame(wday_future),
            pd.DataFrame(month_future),
            pd.DataFrame(event_name_1_future),
            pd.DataFrame(event_type_1_future),
            #pd.DataFrame(event_name_2_future),
            #pd.DataFrame(event_type_2_future),
            pd.DataFrame(yhat_future_cols),
            pd.DataFrame(yhat_past_cols),
            pd.DataFrame(y_past_cols),
            pd.DataFrame(y_future_cols)
        ], axis=1)
        
        df_store = pd.concat([initial_table, future_and_past_shifts], axis=1)
        df_store = df_store[:-28]
        table_with_laggs[store] = df_store.copy()
        table_with_laggs_validation[store] = df_store[:-29].copy()

        
    df_all = (
        pd.concat(table_with_laggs.values())
    )    

    df_all_validation = (
        pd.concat(table_with_laggs_validation.values())
        .drop(columns="date", errors="ignore")
        .dropna()
        .reset_index(drop=True)
    )  

    cat_cols = (['event_name_1','event_type_1','event_name_2','event_type_2','state','store_id']
    +[f'event_name_1_future_{i}' for i in np.arange(1,29)]
    +[f'event_type_1_future_{i}' for i in np.arange(1,29)]
    #+[f'event_name_2_future_{i}' for i in np.arange(1,29)]
    #+[f'event_type_2_future_{i}' for i in np.arange(1,29)]
    )
        
    for col in cat_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))
        df_all_validation[col] = le.fit_transform(df_all_validation[col].astype(str))
        
    return df_all, df_all_validation

def train_regressor(table_with_laggs_validation):
    
    label_cols = [f'y_future_{i}' for i in np.arange(1,29)]

    features_train = table_with_laggs_validation.sample(frac=0.8, random_state=42).reset_index(drop=True)
    labels_train = features_train[label_cols]
    features_train = features_train.drop(columns=label_cols)

    features_test = table_with_laggs_validation.drop(features_train.index).reset_index(drop=True)
    labels_test = features_test[label_cols]
    features_test = features_test.drop(columns=label_cols)

    # -------------------------
    # XGBoost training (one model per horizon)
    # -------------------------
    future_horizons = labels_train.shape[1]
    xgb_models = []
    labels_train_pred_xgb = np.zeros_like(labels_train)
    labels_test_pred_xgb = np.zeros_like(labels_test)

    for i in range(future_horizons):
        model = xgb.XGBRegressor(
            n_estimators=110,
            max_depth=12,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(features_train, labels_train.iloc[:, i])
        xgb_models.append(model)
        
        labels_train_pred_xgb[:, i] = model.predict(features_train)
        labels_test_pred_xgb[:, i] = model.predict(features_test)
        print(str(i)+' out of '+ str(future_horizons))

    # Compute RMSE %
    rmse_train_xgb = np.sqrt(np.mean((labels_train - labels_train_pred_xgb) ** 2, axis=0))
    rmse_test_xgb = np.sqrt(np.mean((labels_test - labels_test_pred_xgb) ** 2, axis=0))

    mean_rmse_train_xgb_pct = rmse_train_xgb.mean() / labels_train.values.mean() * 100
    mean_rmse_test_xgb_pct = rmse_test_xgb.mean() / labels_test.values.mean() * 100

    print(f"XGBoost Train RMSE%: {mean_rmse_train_xgb_pct:.2f}%")
    print(f"XGBoost Test RMSE%: {mean_rmse_test_xgb_pct:.2f}%")

    save_dir = "xgb_models_28d"
    os.makedirs(save_dir, exist_ok=True)

    for i, model in enumerate(xgb_models):
        filepath = os.path.join(save_dir, f"xgb_model_h{i}.pkl")
        joblib.dump(model, filepath)

def load_xgb_models(future_horizons=28):
    loaded_models = []
    save_dir = "xgb_models_28d"

    for i in range(future_horizons):
        filepath = os.path.join(save_dir, f"xgb_model_h{i}.pkl")
        model = joblib.load(filepath)
        loaded_models.append(model)
    
    return loaded_models

def evaluate_forecast_of_cumulative_sales(loaded_models, feature_table_with_laggs):
    features_with_labels_evaluate_by_store = {
        store_id: group.copy()
        for store_id, group in feature_table_with_laggs.groupby("store_id")
    }
    n_horizons = len(loaded_models)
    for store in range(10):
        label_cols = [f'y_future_{i}' for i in np.arange(1,29)]

        # Drop rows with NaN in any of the label columns
        features_with_labels_clean = features_with_labels_evaluate_by_store[store].dropna() #.reset_index(drop=True)

        features_evaluate = features_with_labels_clean.drop(columns=label_cols).drop(columns='date')#.reset_index(drop=True)
        labels_evaluate = features_with_labels_clean[label_cols]

        preds_all = np.column_stack([m.predict(features_evaluate) for m in loaded_models])
        pred_cols = [f"ypred_{i}" for i in range(n_horizons)]
        y_cols = [f"y_{i}" for i in range(n_horizons)]

        eval_df = pd.DataFrame(index=features_evaluate.index)
        eval_df["date"] = features_with_labels_clean["date"]
        eval_df[y_cols] = labels_evaluate.values
        eval_df[pred_cols] = preds_all
        valid_dates = set(eval_df['date'])

        eval_df_new = eval_df[['date', 'y_0', 'ypred_0']].copy()
        eval_df_new['date'] = eval_df_new['date'] + pd.Timedelta(days=1)

        eval_df_new = eval_df_new[eval_df_new['date'].isin(valid_dates)]

        plt.figure(figsize=(12, 4))
        plt.plot(eval_df_new['date'], eval_df_new['y_0'], label='Actual')
        plt.plot(eval_df_new['date'], eval_df_new['ypred_0'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.waitforbuttonpress()

    print(features_with_labels_evaluate_by_store[0])