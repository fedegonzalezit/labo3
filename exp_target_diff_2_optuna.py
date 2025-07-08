import pandas as pd
import numpy as np
import re
import lightgbm as lgb

DATA_FOLDER = "./"

df = pd.read_pickle(DATA_FOLDER + "df_fe_epic_light_grouped.pickle")

TEST_DATE = 33
EXP_FOLDER = "./experiments/test_optuna/"

# create folder if not exists
import os

if not os.path.exists(EXP_FOLDER):
    os.makedirs(EXP_FOLDER)


def save_submission(df_result, filename):
    df_result = df_result.reset_index()
    df_result = df_result[["product_id", "predictions"]]
    df_result.columns = ["product_id", "tn"]
    df_result.to_csv(EXP_FOLDER + filename, index=False)


def get_indexes(df, test_date=TEST_DATE):
    test_index = df.index[df["date_id"] == test_date]
    train_index = df.index[df["date_id"] <= test_date - 2]
    train_scaler_index = df.index[df["date_id"] <= test_date]
    return test_index, train_index, train_scaler_index


numeric_columns = df.select_dtypes(include=["float64", "float32"]).columns
print(numeric_columns)
# transformations = {
#    "tn": [r"tn$", r"cust_request_qty_per_tn$", r"tn_lag_*", r"tn_rolling_mean_*", r"tn_rolling_max_*", r"tn_rolling_min_*", r"tn_.*_vendidas$"],
#    "stock_final": [r"stock_final$"],
#    "cust_request_tn_minus_tn": [r"cust_request_tn_minus_tn$"],
#    "tn_diff_2": [r"tn_diff_*"]
# }
transformations = {
    "tn": [
        r"tn$",
        r"cust_request_qty_per_tn$",
        r"tn_lag_*",
        r"tn_rolling_mean_*",
        r"tn_rolling_max_*",
        r"tn_rolling_min_*",
        r"tn_.*_vendidas$",
        r"tn_agg*",
    ]
    + [r"stock_final$"]
    + [r"cust_request_tn_minus_tn$"]
    + [r"tn_diff_*"],
    "cust_request_qty": [
        r"cust_request_qty$",
        r"cust_request_qty_lag_*",
        r"cust_request_qty_rolling_mean_*",
        r"cust_request_qty_rolling_max_*",
        r"cust_request_qty_rolling_min_*",
        r"cust_request_qty_.*_vendidas$",
        r"cust_request_qty_agg*",
    ]
    + [r"cust_request_qty_diff_*"],
}


def scale_df(df, transformations, train_scaler_index):
    df_scaled = df  # no hago copy intencionalmente
    train_scaler_df = df_scaled.loc[train_scaler_index]

    #prod_stats = train_scaler_df.groupby("product_id")[
    #    list(transformations.keys())
    #].agg(["mean", "std"])
    #print(prod_stats.head())

    #def custom_group_stats(group):
    #    product_id = group.name[1]
    #    row = {"customer_id": group.name[0], "product_id": product_id}
    #    for col in transformations.keys():
    #        std_prod = prod_stats.loc[product_id, (col, "std")]
    #        mean = group[col].mean()
    #        std = max(group[col].std(), std_prod, 1)
    #        row[f"{col}_mean"] = mean
    #        row[f"{col}_std"] = std
    #    return pd.Series(row)

    #group_stats = (
    #    train_scaler_df.groupby(["customer_id", "product_id"])
    #    .apply(custom_group_stats)
    #    .reset_index(drop=True)
    #)

    prod_stats = train_scaler_df.groupby("product_id")[
        list(transformations.keys())
    ].agg(["mean", "std"])
    #print(prod_stats.head())

    def custom_group_stats(group):
        product_id = group.name[1]
        row = {"customer_id": group.name[0], "product_id": product_id}
        for col in transformations.keys():
            std_prod = prod_stats.loc[product_id, (col, "std")]
            mean = group[col].mean()
            std = std_prod
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = mean # uso la media para dividir en vez del std

            #row[f"{col}_std"] = std
        return pd.Series(row)

    group_stats = (
        train_scaler_df.groupby(["customer_id", "product_id"])
        .apply(custom_group_stats)
        .reset_index(drop=True)
    )
    print(group_stats.head())

    # Mergear las stats al df original
    df_scaled = df_scaled.merge(
        group_stats, on=["product_id", "customer_id"], how="left"
    )
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
                scaled_cols[col + "_scaled"] = (df_scaled[col]) / df_scaled[
                    trainer + "_std"
                ]

    # Crear un DataFrame con todas las columnas escaladas
    scaled_df = pd.DataFrame(scaled_cols, index=df_scaled.index)

    # Concatenar de una sola vez
    df_scaled = pd.concat([df_scaled, scaled_df], axis=1)
    aux_cols = [col + "_mean" for col in list(transformations.keys())] + [
        col + "_std" for col in list(transformations.keys())
    ]
    df_scaled = df_scaled.drop(columns=aux_cols)
    return df_scaled, group_stats


def zero_feature(df):
    # agregar una feature categorica que es True is tn es 0
    df["tn_zero"] = df["tn_scaled"] == 0
    # Convertir a tipo categoría
    df["tn_zero"] = df["tn_zero"].astype("category")
    return df


def fill_cat_features(df):
    cat_cols = [col for col in df.columns if df[col].dtype.name == "category"]
    for col in cat_cols:
        if "missing" not in df[col].cat.categories:
            # Agregar la categoría "missing" si no existe
            df[col] = df[col].cat.add_categories("missing")
        df[col] = df[col].fillna("missing")
    return df


def replace_inf_with_nan(df, fillna=True):

    not_cat_cols = [col for col in df.columns if df[col].dtype.name != "category"]
    df[not_cat_cols] = df[not_cat_cols].replace([np.inf, -np.inf], np.nan)
    if fillna:
        df[not_cat_cols] = df[not_cat_cols].fillna(0)
    return df


def drop_single_unique_value_columns(df):
    cols_to_drop = []
    for col in df.columns:
        if col != "customer_id" and df[col].nunique() <= 1:
            cols_to_drop.append(col)
    print("Dropping columns with single unique value:", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
    return df


def get_train_test_df(df, train_index, test_index):
    y_train = df.loc[train_index, "target"].dropna()
    X_train = df.loc[y_train.index].drop(columns=["target", "fecha"])
    y_test = df.loc[test_index, "target"]
    X_test = df.loc[test_index].drop(columns=["target", "fecha"])
    product_ids = pd.read_csv(DATA_FOLDER + "product_id_apredecir201912.txt", sep="\t")[
        "product_id"
    ].tolist()
    X_test = X_test[X_test["product_id"].isin(product_ids)]
    test_index = X_test.index
    return X_train, y_train, X_test, y_test, test_index


def mask_train_zeros(X_train, y_train):

    # Marca los grupos donde tn_scaled es siempre 0
    mask = X_train.groupby(["customer_id", "product_id"])["tn"].transform(
        lambda x: (x == 0).all()
    )
    # Guarda los pares eliminados
    deleted_pairs = X_train.loc[mask, ["customer_id", "product_id"]].drop_duplicates()
    deleted_pairs_set = set(map(tuple, deleted_pairs.values))

    # Elimina las filas
    X_train = X_train[~mask]
    y_train = y_train.loc[X_train.index]
    print("Deleted series", len(deleted_pairs))
    return X_train, y_train, deleted_pairs_set


def delete_low_variance_columns(X_train, X_test, threshold=1e-6):
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    low_var_cols = [col for col in num_cols if X_train[col].std() < threshold]
    # Eliminar customer_id de low_var_cols si existe
    if "customer_id" in low_var_cols:
        low_var_cols.remove("customer_id")
    if low_var_cols:
        print("Eliminando columnas de baja varianza:", low_var_cols)
        X_train = X_train.drop(columns=low_var_cols)
        X_test = X_test.drop(columns=low_var_cols)
    return X_train, X_test


def time_decay_weights(X_train, decay_factor=0.99):
    unique_train_dates = X_train["date_id"].unique()
    # date_weights = {date_id: decay_factor ** (len(unique_dates) - idx - 1) for idx, date_id in enumerate(unique_dates)}
    date_weights = {
        date_id: decay_factor ** (len(unique_train_dates) - idx - 1)
        for idx, date_id in enumerate(unique_train_dates)
    }
    # Map the weights to the DataFrame
    weight = X_train["date_id"].map(date_weights).fillna(1)
    return weight


def predict_test(model, X_test, real_target, deleted_pairs_set, group_stats=None):
    # X_test = df.loc[test_index].drop(columns=["target", "fecha"])
    predictions = model.predict(X_test)
    df_result = X_test[["customer_id", "product_id", "tn_scaled", "tn"]].copy()
    #df_result = X_test[["customer_id", "product_id", "tn"]].copy()
    df_result["predictions_scaled"] = predictions
    #df_result["predictions"] = predictions
    mask_deleted = df_result.apply(
        lambda row: (row["customer_id"], row["product_id"]) in deleted_pairs_set, axis=1
    )
    df_result.loc[mask_deleted, "predictions_scaled"] = 0
    df_result.loc[mask_deleted, "predictions"] = 0

    # Mergeá las stats de tn
    df_result = df_result.merge(
        group_stats[["customer_id", "product_id", "tn_std", "tn_mean"]],
        on=["customer_id", "product_id"],
        how="left",
    )

    df_result.set_index(X_test.index, inplace=True)
    df_result["predictions"] = df_result["predictions_scaled"] * df_result["tn_std"]

    df_result = df_result[["customer_id", "product_id", "predictions"]]
    df_result["target"] = real_target.loc[X_test.index, "tn"]
    return df_result


# agrupo por product_id y sumo todos los target y predictions
def calculate_total_error(df_result, alpha=1.0, iter="", submit=False):
    grouped = (
        df_result.groupby("product_id")
        .agg({"predictions": "sum", "target": "sum"})
        .reset_index()
    )
    grouped["predictions"] = grouped["predictions"] * alpha
    grouped["predictions"] = grouped["predictions"].clip(
        lower=0
    )  # Asegurar que las predicciones no sean negativas
    grouped["abs_error"] = np.abs(grouped["predictions"] - grouped["target"])
    if submit:
        save_submission(grouped, f"submission_iter{iter}_alpha{alpha}.csv")
    total_error = grouped["abs_error"].sum() / grouped["target"].sum()
    return grouped, total_error


def train_lgb_model(
    X_train,
    y_train,
    X_test,
    real_target,
    deleted_pairs_set,
    group_stats=None,
    time_decay_factor=0.995,
    base=10,
    lr=0.125,
    lr_decay=0.999,
    use_callbacks=True,
    lgb_params={
        "num_leaves": 31,
        "tweedie_variance_power": 1.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_data_in_leaf": 5,
        "min_data_in_leaf": 2,
        "min_gain_to_split": 0.0,
        "n_estimators": 2000,
    },
    alpha=1.0,
    min_lr=0.001,
):
    weight = time_decay_weights(X_train, decay_factor=time_decay_factor)
    eps = 1e-2
    tn_weight = np.log1p(X_train["tn"]) / np.log(base)

    weight = weight * tn_weight
    weight = weight.clip(lower=eps)  # Evitar pesos cero
    weight = weight * len(weight) / weight.sum()

    print(weight.describe())

    cat_features = [
        col for col in X_train.columns if X_train[col].dtype.name == "category"
    ]
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features, weight=weight
    )

    # creo callback que se ejecuta cad 200 iteraciones y calcula el total_error
    def total_error_callback(env):
        if env.iteration % 200 == 0 and env.iteration > 0:
            df_result = predict_test(
                env.model, X_test, real_target, deleted_pairs_set, group_stats
            )
            grouped, total_error = calculate_total_error(
                df_result, iter=env.iteration, alpha=alpha
            )
            print(f"Iteration {env.iteration}, Total Error: {total_error:.4f}")
            print(grouped.sort_values(by="abs_error", ascending=False).head(10))

    # create learning_rate scheduler, arranca en 0.1 y cada iteracion baja 0.99 ** iter
    def learning_rate_scheduler(iteration):
        new_lr = lr * (lr_decay**iteration)
        new_lr = max(
            new_lr, min_lr
        )  # Ensure the learning rate does not go below min_lr
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Learning Rate: {new_lr:.6f}")
        return new_lr

    callbacks = [
        lgb.reset_parameter(learning_rate=learning_rate_scheduler),
    ]
    if use_callbacks:
        callbacks += [
            lgb.log_evaluation(period=50),
            total_error_callback,
        ]


    model = lgb.train(
        params={
            "objective": "tweedie",
            "boosting_type": "gbdt",
            "metric": "rmse",
            "force_row_wise": True,
            "verbose": -1,
            "max_bin": 512,
            
            **lgb_params,
        },
        train_set=train_data,
        callbacks=callbacks,
        # valid_sets=[train_data],
    )
    return model


import optuna


def train_model_for_optuna(df):
    test_index, train_index, train_scaler_index = get_indexes(df, test_date=33)
    df, group_stats = scale_df(df, transformations, train_scaler_index)

    df["target"] = df.groupby(["customer_id", "product_id"])["tn_scaled"].shift(-2)
    #df["target"] = df.groupby(["customer_id", "product_id"])["tn"].shift(-2)
    print(df["target"].describe())
    #df = zero_feature(df)
    df = fill_cat_features(df)
    df = replace_inf_with_nan(df, fillna=True)
    real_target = pd.DataFrame(
        df.groupby(["customer_id", "product_id"])["tn"].shift(-2)
    )
    # cat_cols = [col for col in df.columns if df[col].dtype.name == 'category']
    # df = df.drop(columns=cat_cols) # ojo que sacar las columnas categoricas mejora test en 0.01
    df = drop_single_unique_value_columns(df)
    X_train, y_train, X_test, y_test, test_index = get_train_test_df(
        df, train_index, test_index
    )
    X_train, y_train, deleted_pairs_set = mask_train_zeros(X_train, y_train)
    X_train, X_test = delete_low_variance_columns(X_train, X_test, threshold=1e-6)

    # hago optuna
    def objective(trial):
        time_decay_factor = trial.suggest_float("time_decay_factor", 0.95, 1)
        base = trial.suggest_float("base", 1.1, 20)
        lr = trial.suggest_float("lr", 0.05, 0.3)
        min_lr = trial.suggest_float("min_lr", 0.001, 0.05)
        lr_decay = trial.suggest_float("lr_decay", 0.99, 1)
        alpha = trial.suggest_float("alpha", 0.9, 1.1)
        lgb_params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 512),
            "tweedie_variance_power": trial.suggest_float(
                "tweedie_variance_power", 1.1, 1.9
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 0, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 750),
        }

        model = train_lgb_model(
            X_train,
            y_train,
            X_test,
            real_target,
            deleted_pairs_set,
            group_stats,
            time_decay_factor=time_decay_factor,
            base=base,
            lr=lr,
            lr_decay=lr_decay,
            lgb_params=lgb_params,
            use_callbacks=False,  # No usar callbacks en la optimización
            alpha=alpha,
            min_lr=min_lr,
        )
        df_result = predict_test(
            model, X_test, real_target, deleted_pairs_set, group_stats
        )
        grouped, total_error = calculate_total_error(
            df_result, iter="optuna", alpha=alpha, submit=False
        )
        return total_error

    # Crear un estudio de Optuna
    # guardo el estudio en una db
    # optuna.create_study()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="exp_target_diff_2_optuna_log1p_mean_scaled",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )
    # Ejecutar la optimización
    study.optimize(objective, n_trials=40)
    print("Best trial:")
    trial = study.best_trial
    return study.best_params


best_params = train_model_for_optuna(df)

test_index, train_index, train_scaler_index = get_indexes(df, test_date=35)
df, group_stats = scale_df(df, transformations, train_scaler_index)

df["target"] = df.groupby(["customer_id", "product_id"])["tn_scaled"].shift(-2)
#df["target"] = df.groupby(["customer_id", "product_id"])["tn"].shift(-2)
#print(df["target"].describe())
#df = zero_feature(df)
df = fill_cat_features(df)
df = replace_inf_with_nan(df, fillna=True)
real_target = pd.DataFrame(df.groupby(["customer_id", "product_id"])["tn"].shift(-2))
# cat_cols = [col for col in df.columns if df[col].dtype.name == 'category']
# df = df.drop(columns=cat_cols) # ojo que sacar las columnas categoricas mejora test en 0.01
df = drop_single_unique_value_columns(df)
X_train, y_train, X_test, y_test, test_index = get_train_test_df(
    df, train_index, test_index
)
X_train, y_train, deleted_pairs_set = mask_train_zeros(X_train, y_train)
X_train, X_test = delete_low_variance_columns(X_train, X_test, threshold=1e-6)

model = train_lgb_model(
    X_train,
    y_train,
    X_test,
    real_target,
    deleted_pairs_set,
    time_decay_factor=best_params["time_decay_factor"],
    base=best_params["base"],
    lr=best_params["lr"],
    lr_decay=best_params["lr_decay"],
    alpha=best_params["alpha"],
    min_lr=best_params["min_lr"],
    lgb_params={
        "num_leaves": best_params["num_leaves"],
        "tweedie_variance_power": best_params["tweedie_variance_power"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "feature_fraction": best_params["feature_fraction"],
        "bagging_fraction": best_params["bagging_fraction"],
        "bagging_freq": best_params["bagging_freq"],
        "min_child_weight": best_params["min_child_weight"],
        "n_estimators": best_params["n_estimators"],
    },
    group_stats=group_stats
)
df_result = predict_test(model, X_test, real_target, deleted_pairs_set, group_stats)
grouped, total_error = calculate_total_error(
    df_result, iter="final", alpha=best_params["alpha"], submit=True
)
