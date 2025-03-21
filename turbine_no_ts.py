# %%
import os
from datetime import datetime as dt
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as L  # noqa: F401
import seaborn as sns
import torch
from pytorch_lightning.callbacks import (  # noqa: F401
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: F401
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.notebook import tqdm
from xgboost import XGBRegressor

import hephaestus.single_row_models as sr

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load and preprocess the train_dataset (assuming you have a CSV file)
csv_files = glob.glob("data/nox/*.csv")

# Read and combine all files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    filename = Path(file).stem  # Get filename without extension
    temp_df["filename"] = filename
    df_list.append(temp_df)

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)
df = df.rename(columns={"nox": "target"})
# scale the non-target numerical columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
numeric_cols = numeric_cols.drop("target")
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.head()

# %%
single_row_config = sr.SingleRowConfig.generate(df, "target")
train_df, test_df = train_test_split(df.copy(), test_size=0.2, random_state=42)
train_dataset = sr.TabularDS(train_df, single_row_config)
test_dataset = sr.TabularDS(test_df, single_row_config)
model = sr.TabularRegressor(single_row_config, d_model=64, n_heads=4, lr=0.01)
model.predict_step(train_dataset[0:10])

# %%
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1024 * 4,
    shuffle=True,
    collate_fn=sr.training.tabular_collate_fn,
    num_workers=7,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1024 * 4,
    collate_fn=sr.training.tabular_collate_fn,
    num_workers=7,
    persistent_workers=True,
)

# %%
logger_variant_name = "Nox_No_TS"
logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
logger_name = f"{logger_time}_{logger_variant_name}"
print(f"Using logger name: {logger_name}")
logger = TensorBoardLogger(
    "runs",
    name=logger_name,
)
model_summary = ModelSummary(max_depth=3)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
progress_bar = TQDMProgressBar(leave=False)
trainer = L.Trainer(
    max_epochs=200,
    logger=logger,
    callbacks=[early_stopping, progress_bar, model_summary],
    log_every_n_steps=1,
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# %%
y_hat = model.predict_step(train_dataset[0:64])
y = train_dataset[0:64].target
print(y.shape, y_hat.shape)
print(mean_squared_error(y, y_hat))

# %%
y_hat

# %%
1 / 0  # Stop here to avoid running the rest of the code

# %%
model = sr.TabTransformer(train_dataset, n_heads=8).to(train_dataset.device)

batch_size = 3
test_num = train_dataset.X_train_numeric[0:batch_size, :]
test_num_mask = sr.mask_tensor(test_num, model)
test_cat = train_dataset.X_test_categorical[0:batch_size, :]
test_cat_mask = sr.mask_tensor(test_cat, model)
with torch.no_grad():
    x = model(
        test_num_mask,
        test_cat_mask,
        task="mlm",
    )
x[0].shape, x[1].shape

# %%
# Masked Tabular Modeling
base_model_name = "is_model_global2"

model_time = dt.now()
model_time = model_time.strftime("%Y-%m-%dT%H:%M:%S")
model_name = f"{base_model_name}_{model_time}"

model_save_path = "./checkpoints/mtm_models_small.pt"

# %%
model_list = os.listdir("./checkpoints")
if model_save_path.split("/")[-1] in model_list:
    print("Model already exists")
    model_exists = True
else:
    print("Model does not exist")
    model_exists = False

if model_exists:
    model.load_state_dict(torch.load(model_save_path))
else:
    sr.mtm(model, train_dataset, model_name, epochs=100, batch_size=1000, lr=0.001)
    torch.save(model.state_dict(), model_save_path)

# %%
# regression_performance = sr.fine_tune_model(
#     model, train_dataset, model_name="FT100", n_rows=100, epochs=100
# )
# regression_performance

# %%
n_train_rows = [
    # 10,
    100,
    1_000,
    2_000,
    5_000,
    10_000,
    15_000,
    30_000,
    # 40_000,
    train_dataset.X_train.shape[0],
]


# %%
def train_multiple_sizes(pt_model_path, train_dataset, n_train_rows, n_epochs=100):
    model = sr.TabTransformer(train_dataset, n_heads=8).to(train_dataset.device)
    if pt_model_path is not None:
        model.load_state_dict(torch.load(pt_model_path))

    regression_performance = sr.fine_tune_model(
        model,
        train_dataset,
        model_name=f"ft_{n_train_rows}",
        n_rows=n_train_rows,
        epochs=n_epochs,
    )

    return regression_performance


# %%
hephaestus_results_no_pre_train = []
pbar = tqdm(n_train_rows)
for i in pbar:
    pbar.set_description(f"n_rows: {i}")
    loss = train_multiple_sizes(None, train_dataset, i, n_epochs=250)
    hephaestus_results_no_pre_train.append(loss)

# %%
no_pt_df = pd.DataFrame(hephaestus_results_no_pre_train)
no_pt_df["model"] = "Hephaestus No Fine Tune"
no_pt_df

# %%
hephaestus_results = []
pbar = tqdm(n_train_rows)
for i in pbar:
    pbar.set_description(f"n_rows: {i}")
    loss = train_multiple_sizes(model_save_path, train_dataset, i, n_epochs=250)
    hephaestus_results.append(loss)

# %%
hephaestus_df = pd.DataFrame(hephaestus_results)
hephaestus_df["model"] = "Hephaestus"
hephaestus_df

# %%
hephaestus_df.loc[hephaestus_df.n_rows == 1000, "test_loss"].values

# %%
diamonds_data = pd.read_csv("./data/diamonds.csv")

# Encode categorical features using LabelEncoder
label_encoders = {}
categorical_features = ["cut", "color", "clarity"]
for feature in categorical_features:
    le = LabelEncoder()
    diamonds_data[feature] = le.fit_transform(diamonds_data[feature])
    label_encoders[feature] = le

# Split the train_dataset into features (X) and target (y)
X = diamonds_data.drop("price", axis=1)
y = diamonds_data["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the XGBoost regressor
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_regressor.fit(
    X_train[0:batch_size],
    y_train[0:batch_size],
)

# Predict on the test set
y_pred = xgb_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:,.2f}")

# You can also access feature importance scores
# feature_importances = xgb_regressor.feature_importances_
# print("Feature Importance:")
# for feature, importance in zip(X.columns, feature_importances):
#     print(f"{feature}: {importance:.4f}")


# %%
def xgb_tester(train_set_size):
    xgb_regressor = XGBRegressor(n_estimators=120, learning_rate=0.1, random_state=42)
    xgb_regressor.fit(
        X_train[0:train_set_size],
        y_train[0:train_set_size],
    )

    y_pred = xgb_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {"n_rows": train_set_size, "test_loss": mse}


xgb_losses = []
for i in tqdm(n_train_rows):
    mse = xgb_tester(i)
    xgb_losses.append(mse)

# %%
xgb_df = pd.DataFrame(xgb_losses)
xgb_df["model"] = "XGBoost"

xgb_df

# %%
loss_df = pd.concat([hephaestus_df, xgb_df, no_pt_df])  # , no_pt_df
loss_df = loss_df.loc[loss_df["n_rows"] != 10]
loss_df.sample(10)

# %%
# Define the colors for each model
# colors = {"Hephaestus": "blue", "XGBoost": "red"}

# Create a figure and axis object
fig, ax = plt.subplots()

# Loop through each model and plot the test loss as a line
for model, group in loss_df.groupby("model"):
    ax.plot(group["n_rows"], group["test_loss"], label=model)

# Set the axis labels and legend
ax.set_xlabel("Number of Rows")
ax.set_ylabel("Test Loss")
ax.legend()
# set x axis to log scale
ax.set_xscale("log")

# Show the plot
plt.show()

# %%
# Define the colors for each model
# colors = {"Hephaestus": "blue", "XGBoost": "red"}

# Create a figure and axis object
fig, ax = plt.subplots()

# Loop through each model and plot the test loss as a line
for model, group in loss_df.loc[loss_df["model"] != "Hephaestus No Fine Tune"].groupby(
    "model"
):
    ax.plot(group["n_rows"], group["test_loss"], label=model)

# Set the axis labels and legend
ax.set_xlabel("Number of Rows")
ax.set_ylabel("Test Loss")
ax.legend()
# set x axis to log scale
ax.set_xscale("log")

# Show the plot
plt.show()

# %%
# Spread the data to have columns for the loss of each model
# loss_df =
loss_percent_df = loss_df.pivot(
    index="n_rows", columns="model", values="test_loss"
).reset_index()
loss_percent_df["percent_improvement"] = (
    loss_percent_df["XGBoost"] - loss_percent_df["Hephaestus"]
) / loss_percent_df["XGBoost"]

# %%
loss_percent_df

# %%
ax = sns.lineplot(data=loss_percent_df, x="n_rows", y="percent_improvement")
plt.axhline(y=0, color="black", linestyle="--")
ax.set_yticks(loss_percent_df["percent_improvement"].round(2))
ax.set_xticks(loss_percent_df["n_rows"])
# ax.set_xscale("log")
# X lables at 45 degree angle
plt.xticks(rotation=45)
# plt.xlabel
