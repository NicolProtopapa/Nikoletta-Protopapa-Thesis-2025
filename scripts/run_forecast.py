import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator
import torch

# Paths
data_path = "data/full_exchange_rates.csv"
ckpt_path = "lag-llama-model/lag-llama.ckpt"  # adjust to your actual checkpoint file
output_path = "data/forex_forecast.csv"

# Load your data
df = pd.read_csv(data_path)
print("Data loaded, shape:", df.shape)

# Make sure your data is univariate and has a datetime index
# Example: assume df has columns "date" and "value"
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").asfreq("D")  # change "D" if your data has different freq

ts = df["value"].values  # replace "value" with the correct column name

# Create a GluonTS dataset
prediction_length = 30  # how many days ahead you want to predict — change as needed
context_length = 128    # how many historical points to use — tune this

dataset = ListDataset(
    [{"start": df.index[0], "target": ts}],
    freq="D"  # change to your data frequency
)

# Load checkpoint and hyperparameters
ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
est_args = ckpt["hyper_parameters"]["model_kwargs"]

# Create Lag‑Llama estimator
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=est_args["input_size"],
    n_layer=est_args["n_layer"],
    n_embd_per_head=est_args["n_embd_per_head"],
    n_head=est_args["n_head"],
    scaling=est_args["scaling"],
    time_feat=est_args["time_feat"],
)

# Create predictor
predictor = estimator.create_predictor(
    transformation=estimator.create_transformation(),
    module=estimator.create_lightning_module()
)

# Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset, predictor=predictor
)

forecasts = list(forecast_it)
tss = list(ts_it)

# For simplicity: take the first forecast
first_forecast = forecasts[0]

# Convert GluonTS forecast to a pandas-friendly format
import numpy as np

mean_pred = np.mean(first_forecast.samples, axis=0)
dates = pd.date_range(start=df.index[-1] + pd.Timedelta(1, unit="D"), periods=prediction_length)
pred_df = pd.DataFrame({"date": dates, "mean_forecast": mean_pred})

# Save to CSV
pred_df.to_csv(output_path, index=False)
print("Forecast saved to", output_path)