from GroceryRetailSalesData import GroceryRetailSalesData, CumulativeSalesTimeSeriesData

def generate_cumulative_sales_time_series(initial_data: GroceryRetailSalesData):
    result = CumulativeSalesTimeSeriesData()

    for store in initial_data._store_ids:
        df_store_validation = initial_data._df_sales_validation[initial_data._df_sales_validation['store_id'] == store].copy()
        day_cols_validation = [col for col in df_store_validation.columns if col.startswith('d_')]
        df_sum_validation = df_store_validation[day_cols_validation].sum(axis=0).reset_index()
        df_sum_validation.columns = ['d', 'y']
        df_sum_validation = df_sum_validation.merge(initial_data._df_calendar[['d','date']], on='d', how='left')
        result._df_validation[store] = df_sum_validation.copy()

        df_store_evaluation = initial_data._df_sales_evaluation[initial_data._df_sales_evaluation['store_id'] == store].copy()
        day_cols_evaluation = [col for col in df_store_evaluation.columns if col.startswith('d_')]
        df_sum_evaluation = df_store_evaluation[day_cols_evaluation].sum(axis=0).reset_index()
        df_sum_evaluation.columns = ['d', 'y']
        df_sum_evaluation = df_sum_evaluation.merge(initial_data._df_calendar[['d','date']], on='d', how='left')
        result._df_evaluation[store] = df_sum_evaluation.copy()

    return result