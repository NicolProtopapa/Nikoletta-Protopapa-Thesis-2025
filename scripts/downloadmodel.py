from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="time-series-foundation-models/Lag-Llama",
    local_dir="lag-llama-model"
)