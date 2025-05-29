# %%
import polars as pl

df = pl.read_parquet("data/combined_3w_dataset.parquet")
df.head()
# %%
df.shape
# %%
df = df.filter(
    (pl.col("well_name") != "SIMULATED") & (pl.col("well_name") != "UNKNOWN")
)
df = df.select(pl.all().exclude("__index_level_0__"))
df.shape

# %%
df = df.with_columns(
    (pl.col("file_class_label").cast(pl.String) + "_" + pl.col("well_name")).alias(
        "system_id"
    )
)


# %%
unique_system_ids = df.select("system_id").unique().sort("system_id")
system_id_subset = unique_system_ids.sample(n=30, with_replacement=False)


# %%
df = df.join(
    system_id_subset,
    on="system_id",
    how="inner",
)

# %%
# %%
df.write_parquet(
    "data/combined_3w_real_sample.parquet",
    compression="snappy",
)
# %%
