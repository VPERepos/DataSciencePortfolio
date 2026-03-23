import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


class WRMSSEEvaluator:
    def __init__(self, train_df, calendar, prices):
        """
        train_df: MUST be sales_train_evaluation.csv for correct WRMSSE
        calendar: calendar.csv
        prices: sell_prices.csv
        """

        self.train_df = train_df.copy()
        self.calendar = calendar
        self.prices = prices

        self.id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        self.d_cols = [c for c in train_df.columns if c.startswith("d_")]

        # 12 hierarchical aggregation levels (official M5)
        self.levels = [
            [],
            ["state_id"],
            ["store_id"],
            ["cat_id"],
            ["dept_id"],
            ["state_id", "cat_id"],
            ["state_id", "dept_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
            ["item_id"],
            ["item_id", "state_id"],
            ["item_id", "store_id"],
        ]

        self._prepare_data()

    # -----------------------------
    # PREP PIPELINE
    # -----------------------------
    def _prepare_data(self):
        self._build_aggregation()
        self._compute_scales()
        self._compute_weights()

    # -----------------------------
    # AGGREGATION MATRIX
    # -----------------------------
    def _build_aggregation(self):
        sales = self.train_df[self.d_cols].values
        n_series = sales.shape[0]

        agg_matrices = []
        all_ids = []

        for group_cols in self.levels:
            if len(group_cols) == 0:
                mat = np.ones((1, n_series), dtype=np.float32)
                ids = np.array(["Total"])
            else:
                group_ids = self.train_df[group_cols].astype(str).agg("_".join, axis=1)
                unique_groups, group_index = np.unique(group_ids, return_inverse=True)

                mat = np.zeros((len(unique_groups), n_series), dtype=np.float32)
                mat[group_index, np.arange(n_series)] = 1

                ids = unique_groups

            agg_matrices.append(mat)
            all_ids.append(ids)

        self.agg_matrix = np.vstack(agg_matrices)
        self.all_ids = np.concatenate(all_ids)

        self.train_series = self.agg_matrix @ sales

    # -----------------------------
    # SCALE (RMSSE denominator)
    # -----------------------------
    def _compute_scales(self):
        scales = []

        for series in self.train_series:
            # Remove leading zeros (IMPORTANT for M5)
            non_zero_idx = np.argmax(series > 0)
            trimmed = series[non_zero_idx:]

            if len(trimmed) <= 1:
                scales.append(1e-8)
            else:
                diff = np.diff(trimmed)
                scales.append(np.mean(diff ** 2))

        self.scale = np.array(scales)

    # -----------------------------
    # WEIGHTS (Revenue-based)
    # -----------------------------
    def _compute_weights(self):
        last_28 = self.d_cols[-28:]

        # Map d -> week
        d_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weeks = np.array([d_to_week[d] for d in last_28])

        sales = self.train_df.copy()
        sales["key"] = sales["store_id"] + "_" + sales["item_id"]

        price_df = self.prices.copy()
        price_df["key"] = price_df["store_id"] + "_" + price_df["item_id"]

        price_map = price_df.set_index(["key", "wm_yr_wk"])["sell_price"]

        sales_values = sales[last_28].values
        price_matrix = np.zeros_like(sales_values, dtype=np.float32)

        keys = sales["key"].values

        for i, week in enumerate(weeks):
            price_matrix[:, i] = price_map.reindex(
                pd.MultiIndex.from_arrays([keys, [week] * len(keys)])
            ).values

        revenue = sales_values * price_matrix

        # Aggregate revenue correctly
        agg_revenue = self.agg_matrix @ revenue
        weights = agg_revenue.sum(axis=1)

        self.weights = weights / weights.sum()

    # -----------------------------
    # SCORE FUNCTION
    # -----------------------------
    def score(self, preds, actuals):
        """
        preds: numpy array (30490, 28)
        actuals: numpy array (30490, 28)
        """

        preds_agg = self.agg_matrix @ preds
        actuals_agg = self.agg_matrix @ actuals

        mse = np.mean((actuals_agg - preds_agg) ** 2, axis=1)
        rmsse = np.sqrt(mse / self.scale)

        return np.sum(self.weights * rmsse)


# =====================================================
# USAGE EXAMPLE
# =====================================================

# IMPORTANT: use evaluation dataset for correct WRMSSE
# train_df = pd.read_csv("sales_train_evaluation.csv")
# calendar = pd.read_csv("calendar.csv")
# prices = pd.read_csv("sell_prices.csv")

# evaluator = WRMSSEEvaluator(train_df, calendar, prices)

# actuals = train_df.iloc[:, -28:].values  # last 28 days
# preds = np.random.rand(30490, 28)        # replace with your model

# score = evaluator.score(preds, actuals)
# print("WRMSSE:", score)

# Sanity check:
# print(evaluator.score(actuals, actuals))  # should be ~0


#Submission type	        WRMSSE range

#Top 1–2 (winning)	    0.520
#Strong / competitive	0.55–0.60
#Baseline/simple models	0.65–0.80
#Poor / misaligned	    >1.0


#validation of the class

#print(weights_df["weight"].sum()) # should be around 1.0

#rev_check = (sales_last_28 * price_matrix).sum(axis=1)
#revenue_ratio = rev_check / rev_check.sum()

# match bottom-level weight with revenue ratio
#np.allclose(weights_df_bottom, revenue_ratio, atol=1e-5) #Should match almost exactly (tiny float differences are okay).
