import pandas as pd
import numpy as np
import re
import lightgbm as lgb

DATA_FOLDER = "./"

df = pd.read_pickle(DATA_FOLDER+'df_fe_epic_light.pickle')

TEST_DATE = 33
EXP_FOLDER = "./experiments/test_exp/"

# create folder if not exists
import os
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

test_index, train_index, train_scaler_index = get_indexes(df)

numeric_columns = df.select_dtypes(include=['float64', "float32"]).columns
print(numeric_columns)
#transformations = {
#    "tn": [r"tn$", r"cust_request_qty_per_tn$", r"tn_lag_*", r"tn_rolling_mean_*", r"tn_rolling_max_*", r"tn_rolling_min_*", r"tn_.*_vendidas$"],
#    "stock_final": [r"stock_final$"],
#    "cust_request_tn_minus_tn": [r"cust_request_tn_minus_tn$"],
#    "tn_diff_2": [r"tn_diff_*"]
#}
transformations = {
    "tn": [r"tn$", r"cust_request_qty_per_tn$", r"tn_lag_*", r"tn_rolling_mean_*", r"tn_rolling_max_*", r"tn_rolling_min_*", r"tn_.*_vendidas$"] + 
    [r"stock_final$"] + [r"cust_request_tn_minus_tn$"] + [r"tn_diff_*"],
    "cust_request_qty": [r"cust_request_qty_*"]
}


def scale_df(df, transformations):
    df_scaled = df # no hago copy intencionalmente
    train_scaler_df = df_scaled.loc[train_scaler_index]

    prod_stats = (
        train_scaler_df.groupby('product_id')[list(transformations.keys())]
        .agg(['mean', 'std'])
    )

    def custom_group_stats(group):
        product_id = group.name[1]
        row = {'customer_id': group.name[0], 'product_id': product_id}
        for col in transformations.keys():
            nonzero_count = (group[col] != 0).sum()
            if nonzero_count <= 3:
                mean = prod_stats.loc[product_id, (col, 'mean')]
                std = prod_stats.loc[product_id, (col, 'std')]
            else:
                mean = group[col].mean()
                std = group[col].std()
                if std < 1:
                    std = max(group[col].max(), 1)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
        return pd.Series(row)

    group_stats = train_scaler_df.groupby(['customer_id', 'product_id']).apply(custom_group_stats).reset_index(drop=True)


    # Mergear las stats al df original
    df_scaled = df_scaled.merge(group_stats, on=['product_id', "customer_id"], how='left')
    df_scaled = df_scaled.set_index(df.index)

    scaled_cols = {}
    for trainer, regex_cols in transformations.items():
        for col in regex_cols:
            # Usar regex para seleccionar las columnas que coinciden
            # chequear si la columna es un regex
            matching_cols = [c for c in numeric_columns if re.match(col, c)]
            if not matching_cols:
                continue  # Si no hay columnas que coincidan, saltar

            # Calcular la media y desviación estándar para cada 
            print(f"Processing trainer: {trainer} with columns: {matching_cols}")
            for match_col in matching_cols:
                mean_col = f"{trainer}_mean"
                std_col = f"{trainer}_std"

            # Escalar las columnas
            for col in matching_cols:
                #scaled_cols[col + "_scaled"] = (df_scaled[col] - df_scaled[trainer + "_mean"]) / df_scaled[trainer + "_std"]
                scaled_cols[col + "_scaled"] = (df_scaled[col]) / df_scaled[trainer + "_std"]
                # reemplazo nan por 0 e inf por 0
                #scaled_cols[col + "_scaled"] = scaled_cols[col + "_scaled"].replace([np.inf, -np.inf], 0)

    # Crear un DataFrame con todas las columnas escaladas
    scaled_df = pd.DataFrame(scaled_cols, index=df_scaled.index)

    # Concatenar de una sola vez
    df_scaled = pd.concat([df_scaled, scaled_df], axis=1)
    aux_cols = [col + "_mean" for col in list(transformations.keys())] + [col + "_std" for col in list(transformations.keys())]
    df_scaled = df_scaled.drop(columns=aux_cols)
    return df_scaled, group_stats

df, group_stats = scale_df(df, transformations)

df['target'] = df.groupby(['customer_id', 'product_id'])['tn_scaled'].shift(-2)

print(df["target"].describe())

real_target = pd.DataFrame(df.groupby(['customer_id', 'product_id'])['tn'].shift(-2))


def predict_test(model, X_test, real_target, test_index):
    X_test = df_scaled.loc[test_index].drop(columns=["target", "fecha"])
    predictions = model.predict(X_test)
    df_result = X_test[['customer_id', 'product_id', "tn_scaled", "tn"]].copy()
    df_result['predictions_scaled'] = predictions

    # Mergeá las stats de tn
    df_result = df_result.merge(
        group_stats[['customer_id', 'product_id', "tn_std", "tn_mean"]],
        #group_stats[['product_id', 'tn_diff_2_mean', 'tn_diff_2_std', "tn_std", "tn_mean"]],
        #group_stats[['customer_id', 'product_id', 'tn_diff_2_median', 'tn_diff_2_iqr', "tn_iqr", "tn_median"]],
        on=['customer_id', 'product_id'],
        #on=['product_id'],
        how='left'
    )
    df_result.set_index(test_index, inplace=True)
    #df_result['predictions'] = df_result['predictions_scaled'] * df_result['tn_std'] + df_result['tn_mean']
    df_result['predictions'] = df_result['predictions_scaled'] * df_result['tn_std']
    #df_result['predictions'] = df_result['predictions_scaled'] * df_result['tn_std'] + df_result['tn_mean']
    # hago la inversa de la diferencia con tn
    #df_result["predictions"] = df_result['predictions'] + df_result['tn']
    # Invertí el escalado

    # multiplico predictions por predictions_df["predictions_binary"] que es el clasificador de 0s
    #df_result["predictions"] = df_result["predictions"] * predictions_df["predictions_binary"]

    df_result = df_result[['customer_id', 'product_id', 'predictions']]
    df_result["target"] = real_target.loc[test_index, "tn"]
    return df_result


# agrupo por product_id y sumo todos los target y predictions
def calculate_total_error(df_result, alpha=1.0, iter=""):
    grouped = df_result.groupby('product_id').agg({
        'predictions': 'sum',
        'target': 'sum'
    }).reset_index()
    grouped['predictions'] = grouped['predictions'] * alpha
    grouped['abs_error'] = np.abs(grouped['predictions'] - grouped['target'])
    save_submission(grouped, f"submission_iter{iter}.csv")
    total_error = grouped['abs_error'].sum() / grouped['target'].sum()
    return grouped, total_error


y_train = df.loc[train_index, 'target'].dropna()
X_train = df.loc[y_train.index].drop(columns=["target", "fecha"])
y_test = df.loc[test_index, 'target']
X_test = df.loc[test_index].drop(columns=["target", "fecha"])
product_ids = pd.read_csv(DATA_FOLDER+"product_id_apredecir201912.txt", sep="\t")["product_id"].tolist()
X_test = X_test[X_test['product_id'].isin(product_ids)]
test_index = X_test.index
cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
#val_data = lgb.Dataset(y_test.loc[y_test.dropna().index], label=y_test.dropna(), reference=train_data)

X_test = df.loc[test_index].drop(columns=["target", "fecha"])
# creo callback que se ejecuta cad 200 iteraciones y calcula el total_error
def total_error_callback(env):
    if env.iteration % 200 == 0 and env.iteration > 0:
        df_result = predict_test(env.model, X_test, real_target, test_index)
        grouped, total_error = calculate_total_error(df_result, iter=env.iteration)
        print(f"Iteration {env.iteration}, Total Error: {total_error:.4f}")
        print(grouped.sort_values(by='abs_error', ascending=False).head(10))

# create learning_rate scheduler, arranca en 0.1 y cada iteracion baja 0.99 ** iter
def learning_rate_scheduler(iteration):
    min_lr = 0.001
    new_lr = 0.1 * (0.999 ** iteration)
    new_lr = max(new_lr, min_lr)  # Ensure the learning rate does not go below min_lr
    return new_lr


def num_leaves_scheduler(iteration):
    # Reduce the number of leaves as the iterations increase
    return max(17, 512 - iteration )  # Ensure it doesn't go below 31

callbacks = [
    lgb.log_evaluation(period=50),
    total_error_callback, 
    lgb.reset_parameter(learning_rate=learning_rate_scheduler)
]
# poisson: 0.38
# huber: 0.3612
# fair: 0.39
# quantile: 2??
# mape: 0.5 - no sirve con labels menores a 1
# gamma (esta deberia usarlo si hago transformacion log):  da errores
# tweedie: 0.3492 (antes de overffitear)
model = lgb.train(
    params={
        'objective': 'tweedie',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'num_leaves': 52,
        'learning_rate': 0.05,
        'feature_fraction': 0.2,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        #'max_depth': 10,
        "max_bin": 512,
        "verbose": 0
    },
    train_set=train_data,
    num_boost_round=2000,
    callbacks=callbacks,
    valid_sets=[train_data],
    #early_stopping_rounds=50
)

df_result = predict_test(model, df_scaled, real_target, test_index)
df_result["abs_error"] = np.abs(df_result['predictions'] - df_result['target'])
#df_result["predictions_binary"] = predictions_df["predictions_binary"]
print(df_result.head())
grouped, total_error = calculate_total_error(df_result, 0.9)
print("Total Error:", total_error)
grouped, total_error = calculate_total_error(df_result, 1)
print("Total Error:", total_error)
grouped, total_error = calculate_total_error(df_result, 1.1)
print("Total Error: (alfa=1.1)", total_error)
print(grouped.sort_values(by='abs_error', ascending=False).head(10))
