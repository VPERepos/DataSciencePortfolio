import matplotlib.pyplot as plt

def plot_prophet_forecasts_validation(models, forecasts, store_ids):
    for store in store_ids:
        fig = models._df_validation_models[store].plot(forecasts._df_validation[store])
        plt.title(f"Prophet Prediction for Store {store}")
        plt.tight_layout()

    plt.show()
    plt.waitforbuttonpress()

def plot_price_indexes(price_indexes_per_store, store_ids):
    for store in store_ids:
        fig, ax = plt.subplots()
        ax.plot(price_indexes_per_store[store].to_numpy())
        ax.set_xlabel("Time index")
        ax.set_ylabel("Normalized Index Value")
        ax.set_title(f"Price Indexes for Store {store}")

        fig.tight_layout()


    plt.show()
    plt.waitforbuttonpress()

def plot_cumulative_sales_preds_vs_actual(eval_df_by_stores):
    store_ids = list(eval_df_by_stores.keys())
    for store in store_ids:
        plt.figure(figsize=(12, 4))
        plt.plot(eval_df_by_stores[store]['date'], eval_df_by_stores[store]['y_0'], label='Actual')
        plt.plot(eval_df_by_stores[store]['date'], eval_df_by_stores[store]['ypred_0'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Actual vs Predicted {store}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.show()
    plt.waitforbuttonpress()
        