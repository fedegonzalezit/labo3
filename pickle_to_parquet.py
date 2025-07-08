# read a pickle and save it as parquet
import pandas as pd
df_pickle = pd.read_pickle("df_fe_epic_light_best_customers.pickle")
df_pickle.to_parquet("df_fe_epic_light_best_customers.parquet")
