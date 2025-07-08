import pandas as pd
import numpy as np
import re
import xgboost as xgb
import os

DATA_FOLDER = "./"

df = pd.read_pickle(DATA_FOLDER+'df_fe_epic_light.pickle')

TEST_DATE = 33
EXP_FOLDER = "./experiments/test_exp/"

if not os.path.exists(EXP_FOLDER):
    os.makedirs(EXP_FOLDER)

def save_submission(df_result, filename):
    df_result = df_result.reset_index()
    df_result = df_result[['product_id', 'predictions']]
    df_result.columns = ["product_id", "tn"]
    df_result.to_csv(EXP_FOLDER + filename, index=False)

def get_indexes(df):
    test_index = df.index[df['date_id'] == TEST_DATE]
    train_index = df.index[df['date_id'] <= TEST_DATE-2]
    train_scaler_index = df.index[df['date_id'] <= TEST_DATE]
    return test_index, train_index, train_scaler_index

test_index, train_index, train_scaler_index = get_indexes(df)

numeric_columns = df.select_dtypes(include=['float64', "float32"]).columns
transformations = {
    "tn": [r"tn$", r"cust_request_qty_per_tn$", r"tn_lag_*", r"tn_rolling_mean_*", r"tn_rolling_max_*", r"tn_rolling_min_*", r"tn_.*_vendidas$"] + 
    [r"stock_final$"] + [r"cust_request_tn_minus_tn$"] + [r"tn_diff_*"],
    "cust_request_qty": [r"cust_request_qty$", r"cust_request_qty_lag_*", r"cust_request_qty_rolling_mean_*", r"cust_request_qty_rolling_max_*", r"cust_request_qty_rolling_min_*", r"cust_request_qty_.*_vendidas$"] + [r"cust_request_qty_diff_*"],
}

def scale_df(df, transformations):
    df_scaled = df
    train_scaler_df = df_scaled.loc[train_scaler_index]
    prod_stats = (
        train_scaler_df.groupby('product_id')[list(transformations.keys())]
        .agg(['mean', "std"])
    )
    def custom_group_stats(group):
        product_id = group.name[1]
        row = {'customer_id': group.name[0], 'product_id': product_id}
        for col in transformations.keys():
            std_prod = prod_stats.loc[product_id, (col, 'std')]
            mean = group[col].mean()
            std = max(group[col].std(), std_prod, 1)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
        return pd.Series(row)
    group_stats = train_scaler_df.groupby(['customer_id', 'product_id']).apply(custom_group_stats).reset_index(drop=True)
    df_scaled = df_scaled.merge(group_stats, on=['product_id', "customer_id"], how='left')
    df_scaled = df_scaled.set_index(df.index)
    scaled_cols = {}
    for trainer, regex_cols in transformations.items():
        for col in regex_cols:
            matching_cols = [c for c in numeric_columns if re.match(col, c)]
            if not matching_cols:
                continue
            for col in matching_cols:
                scaled_cols[col + "_scaled"] = (df_scaled[col]) / df_scaled[trainer + "_std"]
    scaled_df = pd.DataFrame(scaled_cols, index=df_scaled.index)
    df_scaled = pd.concat([df_scaled, scaled_df], axis=1)
    aux_cols = [col + "_mean" for col in list(transformations.keys())] + [col + "_std" for col in list(transformations.keys())]
    df_scaled = df_scaled.drop(columns=aux_cols)
    return df_scaled, group_stats

df, group_stats = scale_df(df, transformations)

df['target'] = df.groupby(['customer_id', 'product_id'])['tn_scaled'].shift(-2)
df['tn_zero'] = df['tn_scaled'] == 0
df['tn_zero'] = df['tn_zero'].astype('category')

print(df["target"].describe())

real_target = pd.DataFrame(df.groupby(['customer_id', 'product_id'])['tn'].shift(-2))

df = df.replace([np.inf, -np.inf], np.nan)

y_train = df.loc[train_index, 'target'].dropna()
X_train = df.loc[y_train.index].drop(columns=["target", "fecha"])



mask = X_train.groupby(['customer_id', 'product_id'])['tn_scaled'].transform(lambda x: (x == 0).all())
deleted_pairs = X_train.loc[mask, ['customer_id', 'product_id']].drop_duplicates()
deleted_pairs_set = set(map(tuple, deleted_pairs.values))
X_train = X_train[~mask]
y_train = y_train.loc[X_train.index]
print("Deleted series", len(deleted_pairs))

y_test = df.loc[test_index, 'target']
X_test = df.loc[test_index].drop(columns=["target", "fecha"])
product_ids = pd.read_csv(DATA_FOLDER+"product_id_apredecir201912.txt", sep="\t")["product_id"].tolist()
X_test = X_test[X_test['product_id'].isin(product_ids)]
test_index = X_test.index
cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

def time_decay_weights(X_train, decay_factor=0.99):
    unique_train_dates = X_train['date_id'].unique()
    date_weights = {date_id: decay_factor ** (len(unique_train_dates) - idx - 1) for idx, date_id in enumerate(unique_train_dates)}
    weight = X_train['date_id'].map(date_weights).fillna(1)
    return weight

weight = time_decay_weights(X_train, decay_factor=0.995)
print(weight.describe())
tn_weight = np.log1p(X_train["tn"])
weight = weight * tn_weight

# XGBoost: convertir cat features a category codes
for col in cat_features:
    X_train[col] = X_train[col].cat.codes
    X_test[col] = X_test[col].cat.codes



X_test = X_test.replace([np.inf, -np.inf], np.nan)

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weight)
dtest = xgb.DMatrix(X_test)

def predict_test(model, X_test, real_target, test_index):
    X_test = df.loc[test_index].drop(columns=["target", "fecha"])
    for col in cat_features:
        X_test[col] = X_test[col].cat.codes
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    df_result = X_test[['customer_id', 'product_id', "tn_scaled", "tn"]].copy()
    df_result['predictions_scaled'] = predictions
    mask_deleted = df_result.apply(lambda row: (row['customer_id'], row['product_id']) in deleted_pairs_set, axis=1)
    df_result.loc[mask_deleted, 'predictions_scaled'] = 0
    df_result = df_result.merge(
        group_stats[['customer_id', 'product_id', "tn_std", "tn_mean"]],
        on=['customer_id', 'product_id'],
        how='left'
    )
    df_result.set_index(test_index, inplace=True)
    df_result['predictions'] = df_result['predictions_scaled'] * df_result['tn_std']
    df_result = df_result[['customer_id', 'product_id', 'predictions']]
    df_result["target"] = real_target.loc[test_index, "tn"]
    return df_result

def calculate_total_error(df_result, alpha=1.0, iter=""):
    grouped = df_result.groupby('product_id').agg({
        'predictions': 'sum',
        'target': 'sum'
    }).reset_index()
    grouped['predictions'] = grouped['predictions'] * alpha
    grouped["predictions"] = grouped["predictions"].clip(lower=0)
    grouped['abs_error'] = np.abs(grouped['predictions'] - grouped['target'])
    save_submission(grouped, f"submission_iter{iter}_alpha{alpha}.csv")
    total_error = grouped['abs_error'].sum() / grouped['target'].sum()
    return grouped, total_error

# XGBoost learning rate scheduler and callback
class LearningRateScheduler:
    def __init__(self, base_lr=0.125, decay=0.999, min_lr=0.001):
        self.base_lr = base_lr
        self.decay = decay
        self.min_lr = min_lr
    def __call__(self, num_round):
        lr = max(self.base_lr * (self.decay ** num_round), self.min_lr)
        if num_round % 50 == 0:
            print(f"Iteration {num_round}, Learning Rate: {lr:.6f}")
        return lr

lr_scheduler = LearningRateScheduler()

class TotalErrorCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # epoch == env.iteration
        if epoch % 200 == 0 and epoch > 0:
            df_result = predict_test(model, X_test, real_target, test_index)
            grouped, total_error = calculate_total_error(df_result, iter=epoch)
            print(f"Iteration {epoch}, Total Error: {total_error:.4f}")
            print(grouped.sort_values(by='abs_error', ascending=False).head(10))
        return False  # return True to stop training

# En tu lista de callbacks:
callbacks = [
    xgb.callback.LearningRateScheduler(lr_scheduler),
    TotalErrorCallback()
]

# XGBoost training
params = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.1,
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'max_bin': 512,
    'max_leaves': 31,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 100,
    'verbosity': 1,
}

num_boost_round = 2000
evals_result = {}


bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, 'train')],
    callbacks=callbacks,
    evals_result=evals_result
)

df_result = predict_test(bst, df, real_target, test_index)
df_result["abs_error"] = np.abs(df_result['predictions'] - df_result['target'])
print(df_result.head())
grouped, total_error = calculate_total_error(df_result, 0.9)
print("Total Error: (alpha=0.9)", total_error)
grouped, total_error = calculate_total_error(df_result, 1)
print("Total Error: (alpha=1)", total_error)
grouped, total_error = calculate_total_error(df_result, 1.1)
print("Total Error: (alfa=1.1)", total_error)
print(grouped.sort_values(by='abs_error', ascending=False).head(10))