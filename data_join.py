# %%
import os
import pandas as pd
# %%
data_dir = "data/beam_data"
files = os.listdir(data_dir)

files = [f for f in files if f != "transactions.csv"]
# %%
files
# %%
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
dfs = pd.concat(dfs)
# %%
trans_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
# %%
df = pd.merge(dfs