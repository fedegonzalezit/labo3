# read a pickle and save it as parquet
import pandas as pd
df_pickle = pd.read_pickle("df_fe_for_ensamble_best_customers_0c.pickle")
df_pickle.to_parquet("df_fe_for_ensamble_best_customers_0c.parquet")
