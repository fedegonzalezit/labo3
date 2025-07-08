import pandas as pd
import numpy as np
import re
import lightgbm as lgb

DATA_FOLDER = "./"

df = pd.read_pickle(DATA_FOLDER+'df_fe_super_hard_grouped.pickle')

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

def get_indexes(df, test_date=TEST_DATE):
    test_index = df.index[df['date_id'] == test_date]
    train_index = df.index[df['date_id'] <= test_date-2]
    train_scaler_index = df.index[df['date_id'] <= test_date]
    return test_index, train_index, train_scaler_index


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
    "cust_request_qty": [r"cust_request_qty$", r"cust_request_qty_lag_*", r"cust_request_qty_rolling_mean_*", r"cust_request_qty_rolling_max_*", r"cust_request_qty_rolling_min_*", r"cust_request_qty_.*_vendidas$"] + [r"cust_request_qty_diff_*"],
}


def scale_df(df, transformations, train_scaler_index):
    df_scaled = df # no hago copy intencionalmente
    train_scaler_df = df_scaled.loc[train_scaler_index]

    prod_stats = (
        train_scaler_df.groupby('product_id')[list(transformations.keys())]
        .agg(['mean', "std"])
    )
    print(prod_stats.head())

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
            # Escalar las columnas
            for col in matching_cols:
                scaled_cols[col + "_scaled"] = (df_scaled[col]) / df_scaled[trainer + "_std"]

    # Crear un DataFrame con todas las columnas escaladas
    scaled_df = pd.DataFrame(scaled_cols, index=df_scaled.index)

    # Concatenar de una sola vez
    df_scaled = pd.concat([df_scaled, scaled_df], axis=1)
    aux_cols = [col + "_mean" for col in list(transformations.keys())] + [col + "_std" for col in list(transformations.keys())]
    df_scaled = df_scaled.drop(columns=aux_cols)
    return df_scaled, group_stats


test_index, train_index, train_scaler_index = get_indexes(df)
df, group_stats = scale_df(df, transformations, train_scaler_index)

df['target'] = df.groupby(['customer_id', 'product_id'])['tn_scaled'].shift(-2)

# agregar una feature categorica que es True is tn es 0
df['tn_zero'] = df['tn_scaled'] == 0
# Convertir a tipo categoría
df['tn_zero'] = df['tn_zero'].astype('category')

print(df["target"].describe())
cat_cols = [col for col in df.columns if df[col].dtype.name == 'category']
for col in cat_cols:
    if "missing" not in df[col].cat.categories:
        # Agregar la categoría "missing" si no existe
        df[col] = df[col].cat.add_categories("missing")
    df[col] = df[col].fillna("missing")

not_cat_cols = [col for col in df.columns if df[col].dtype.name != 'category']
df[not_cat_cols] = df[not_cat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

real_target = pd.DataFrame(df.groupby(['customer_id', 'product_id'])['tn'].shift(-2))


# para probar, saco las columnas categoricas de df
df = df.drop(columns=cat_cols)
# elimino todas las columnas que son iguales para todas las rows (excepto customer_id)

cols_to_drop = []
for col in df.columns:
    if col != "customer_id" and df[col].nunique() <= 1:
        cols_to_drop.append(col)
print("Dropping columns with single unique value:", cols_to_drop)
df = df.drop(columns=cols_to_drop)

y_train = df.loc[train_index, 'target'].dropna()
X_train = df.loc[y_train.index].drop(columns=["target", "fecha"])


# Marca los grupos donde tn_scaled es siempre 0
mask = X_train.groupby(['customer_id', 'product_id'])['tn_scaled'].transform(lambda x: (x == 0).all())
# Guarda los pares eliminados
deleted_pairs = X_train.loc[mask, ['customer_id', 'product_id']].drop_duplicates()
deleted_pairs_set = set(map(tuple, deleted_pairs.values))

# Elimina las filas
X_train = X_train[~mask]
y_train = y_train.loc[X_train.index]
print("Deleted series", len(deleted_pairs))



y_test = df.loc[test_index, 'target']
X_test = df.loc[test_index].drop(columns=["target", "fecha"])
product_ids = pd.read_csv(DATA_FOLDER+"product_id_apredecir201912.txt", sep="\t")["product_id"].tolist()
X_test = X_test[X_test['product_id'].isin(product_ids)]
test_index = X_test.index
cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

num_cols = X_train.select_dtypes(include=[np.number]).columns
low_var_cols = [col for col in num_cols if X_train[col].std() < 1e-6]
# elimino customer_id de low_var_cols si existe
if "customer_id" in low_var_cols:
    low_var_cols.remove("customer_id")
if low_var_cols:
    print("Eliminando columnas de baja varianza:", low_var_cols)
    X_train = X_train.drop(columns=low_var_cols)
    X_test = X_test.drop(columns=low_var_cols)
#X_train.drop(columns=["customer_id"], inplace=True)  # Eliminar customer_id para evitar problemas con el índice
#X_test.drop(columns=["customer_id"], inplace=True)  # Eliminar customer_id para evitar problemas con el índice

def time_decay_weights(X_train, decay_factor=0.99):
        unique_train_dates = X_train['date_id'].unique()
        #date_weights = {date_id: decay_factor ** (len(unique_dates) - idx - 1) for idx, date_id in enumerate(unique_dates)}
        date_weights = {date_id: decay_factor ** (len(unique_train_dates) - idx - 1) for idx, date_id in enumerate(unique_train_dates)}
        # Map the weights to the DataFrame
        weight = X_train['date_id'].map(date_weights).fillna(1)
        return weight


weight = time_decay_weights(X_train, decay_factor=0.995)
print(weight.describe())

base = 5
tn_weight = np.log((X_train["tn"]+1)) / np.log(base)
weight = weight * tn_weight

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, weight=weight)
#val_data = lgb.Dataset(y_test.loc[y_test.dropna().index], label=y_test.dropna(), reference=train_data)



def predict_test(model, X_test, real_target, test_index):
    #X_test = df.loc[test_index].drop(columns=["target", "fecha"])
    predictions = model.predict(X_test)
    df_result = X_test[['customer_id', 'product_id', "tn_scaled", "tn"]].copy()
    df_result['predictions_scaled'] = predictions 
    mask_deleted = df_result.apply(lambda row: (row['customer_id'], row['product_id']) in deleted_pairs_set, axis=1)
    df_result.loc[mask_deleted, 'predictions_scaled'] = 0

    # Mergeá las stats de tn
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


# agrupo por product_id y sumo todos los target y predictions
def calculate_total_error(df_result, alpha=1.0, iter=""):
    grouped = df_result.groupby('product_id').agg({
        'predictions': 'sum',
        'target': 'sum'
    }).reset_index()
    grouped['predictions'] = grouped['predictions'] * alpha
    grouped["predictions"] = grouped["predictions"].clip(lower=0)  # Asegurar que las predicciones no sean negativas
    grouped['abs_error'] = np.abs(grouped['predictions'] - grouped['target'])
    save_submission(grouped, f"submission_iter{iter}_alpha{alpha}.csv")
    total_error = grouped['abs_error'].sum() / grouped['target'].sum()
    return grouped, total_error



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
    new_lr = 0.125 * (0.999 ** iteration)
    new_lr = max(new_lr, min_lr)  # Ensure the learning rate does not go below min_lr
    if iteration % 50 == 0:
        print(f"Iteration {iteration}, Learning Rate: {new_lr:.6f}")
    return new_lr


callbacks = [
    lgb.log_evaluation(period=50),
    total_error_callback, 
    lgb.reset_parameter(learning_rate=learning_rate_scheduler)
]


model = lgb.train(
    params={
        'objective': 'tweedie',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'num_leaves': 31,
        "tweedie_variance_power":1.1,
        "force_row_wise":True,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'verbose': -1,
        "max_bin": 512,
        "verbose": 0,
        "min_data_in_leaf": 5,
        #"linear_tree": True,
        "min_data_in_leaf": 2,
        "min_gain_to_split": 0.0,
    },
    train_set=train_data,
    num_boost_round=2000,
    callbacks=callbacks,
    valid_sets=[train_data],
)

df_result = predict_test(model, X_test, real_target, test_index)
df_result["abs_error"] = np.abs(df_result['predictions'] - df_result['target'])
#df_result["predictions_binary"] = predictions_df["predictions_binary"]
print(df_result.head())
grouped, total_error = calculate_total_error(df_result, 0.9)
print("Total Error: (alpha=0.9)", total_error)
grouped, total_error = calculate_total_error(df_result, 1)
print("Total Error: (alpha=1)", total_error)
grouped, total_error = calculate_total_error(df_result, 1.1)
print("Total Error: (alfa=1.1)", total_error)
print(grouped.sort_values(by='abs_error', ascending=False).head(10))
