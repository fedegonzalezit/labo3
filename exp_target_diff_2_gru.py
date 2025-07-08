import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

DATA_FOLDER = "./"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_pickle(DATA_FOLDER+'df_fe_epic_light_10products.pickle')

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
# Convert categorical features to codes
cat_features = [col for col in df.columns if df[col].dtype.name == 'category']

for col in cat_features:
    df[col] = df[col].cat.codes
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

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

def time_decay_weights(X_train, decay_factor=0.99):
    unique_train_dates = X_train['date_id'].unique()
    date_weights = {date_id: decay_factor ** (len(unique_train_dates) - idx - 1) for idx, date_id in enumerate(unique_train_dates)}
    weight = X_train['date_id'].map(date_weights).fillna(1)
    return weight

weight = time_decay_weights(X_train, decay_factor=0.995)
print(weight.describe())
tn_weight = np.log1p(X_train["tn"])
weight = weight * tn_weight



SEQ_LEN = 12

# --- Sequence preparation ---
def create_sequences_full(df, y_col, w_col, group_cols=['customer_id', 'product_id'], seq_len=SEQ_LEN):
    # Supongamos que tus columnas clave son:
    all_date_ids = np.arange(df['date_id'].min(), df['date_id'].max() + 1)
    # Generar MultiIndex con todos los customer_id, product_id y date_id posibles
    full_index = pd.MultiIndex.from_product(
        [df[c].unique() for c in group_cols] + [all_date_ids],
        names=group_cols + ['date_id']
    )
    # Reindexar el DataFrame
    df = df.set_index(group_cols + ['date_id']).reindex(full_index).reset_index()
    # Rellenar los valores faltantes con 0 (o el valor que prefieras)
    df = df.fillna(0)
    Xs, ys, ws, last_date_ids, last_indices = [], [], [], [], []
    groups = df.groupby(group_cols)
    for _, group in groups:
        group = group.sort_values('date_id')
        values = group.drop(columns=[y_col]).values
        targets = group[y_col].values
        wgts = group[w_col].values if w_col is not None else np.ones(len(group))
        for i in range(len(group) - seq_len + 1):
            Xs.append(values[i:i+seq_len])
            ys.append(targets [i+seq_len-1])
            ws.append(wgts[i+seq_len-1])
    return np.array(Xs), np.array(ys), np.array(ws)

# Antes de split, asegurate de tener las columnas necesarias
df['weight'] = time_decay_weights(df, decay_factor=0.995) * np.log1p(df["tn"]+1)

# Solo columnas numéricas para las secuencias
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("Dropping non-numeric columns for sequence creation:", list(non_numeric_cols))
    df = df.drop(columns=non_numeric_cols)

# Crea secuencias sobre TODO el df
X_seq, y_seq, w_seq = create_sequences_full(
    df, y_col='target', w_col='weight', seq_len=SEQ_LEN
)

last_date_ids = df.groupby(['customer_id', 'product_id'])['date_id'].max().values

# Ahora separá en train/test según el último date_id de la secuencia
train_mask = last_date_ids <= TEST_DATE - 2
test_mask = last_date_ids == TEST_DATE

X_train_seq, y_train_seq, w_train_seq = X_seq[train_mask], y_seq[train_mask], w_seq[train_mask]
X_test_seq, y_test_seq, w_test_seq = X_seq[test_mask], y_seq[test_mask], w_seq[test_mask]

class SequenceDataset(Dataset):
    def __init__(self, X, y, w=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32) if w is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.w is not None:
            return self.X[idx], self.y[idx], self.w[idx]
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train_seq, y_train_seq, w_train_seq)
test_dataset = SequenceDataset(X_test_seq, y_test_seq)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

input_size = X_train_seq.shape[2]
model = GRURegressor(input_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='none')

# --- Learning rate scheduler (callback style) ---
class LearningRateScheduler:
    def __init__(self, base_lr=0.125, decay=0.999, min_lr=0.001):
        self.base_lr = base_lr
        self.decay = decay
        self.min_lr = min_lr
    def __call__(self, epoch):
        lr = max(self.base_lr * (self.decay ** epoch), self.min_lr)
        if epoch % 50 == 0:
            print(f"Iteration {epoch}, Learning Rate: {lr:.6f}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

lr_scheduler = LearningRateScheduler()

# --- Custom callback for evaluation and logging ---
def predict_test(model, X_test, real_target, test_index):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)
    # Prepare results
    test_indices = []
    groups = X_test.groupby(['customer_id', 'product_id'])
    for _, group in groups:
        group = group.sort_values('date_id')
        for i in range(len(group) - SEQ_LEN + 1):
            test_indices.append(group.index[i+SEQ_LEN-1])
    test_indices = np.array(test_indices)
    df_result = X_test.loc[test_indices, ['customer_id', 'product_id', "tn_scaled", "tn"]].copy()
    df_result['predictions_scaled'] = preds
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
    df_result["target"] = real_target.loc[test_indices, "tn"].values
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

class TotalErrorCallback:
    def __init__(self, interval=200):
        self.interval = interval
    def __call__(self, epoch, model):
        if epoch % self.interval == 0 and epoch > 0:
            df_result = predict_test(model, X_test, real_target, test_index)
            grouped, total_error = calculate_total_error(df_result, iter=epoch)
            print(f"Iteration {epoch}, Total Error: {total_error:.4f}")
            print(grouped.sort_values(by='abs_error', ascending=False).head(10))

total_error_callback = TotalErrorCallback(interval=1)

# --- Training loop with callbacks ---
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb, wb in train_loader:
        xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss = (loss * wb).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    lr_scheduler(epoch)
    total_error_callback(epoch, model)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader.dataset):.4f}")

# --- Final prediction and evaluation ---
df_result = predict_test(model, X_test, real_target, test_index)
df_result["abs_error"] = np.abs(df_result['predictions'] - df_result['target'])
print(df_result.head())
grouped, total_error = calculate_total_error(df_result, 0.9)
print("Total Error: (alpha=0.9)", total_error)
grouped, total_error = calculate_total_error(df_result, 1)
print("Total Error: (alpha=1)", total_error)
grouped, total_error = calculate_total_error(df_result, 1.1)
print("Total Error: (alfa=1.1)", total_error)
print(grouped.sort_values(by='abs_error', ascending=False).head(10))