# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hephaestus is a PyTorch-based neural network framework that applies transformer-style attention mechanisms to tabular data. It's designed to handle mixed numeric and categorical data with built-in support for missing values.

## Development Commands

### Installation
```bash
# Install in development mode
uv pip install -e .

# For Lambda Labs environments
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install icecream ipywidgets pyarrow seaborn tqdm transformers
```

### Code Quality
```bash
# Run linter and formatter
ruff check . --fix
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run tests with pytest
pytest
```

## Architecture Overview

The codebase implements a modular architecture for applying attention to tabular data:

1. **Core Models** (`hephaestus/timeseries_models/`):
   - `models.py`: Implements FeedForwardNetwork and attention layers
   - `encoder_decoder.py`: TimeSeriesDecoder with configurable encoder/decoder layers
   - `multihead_attention.py`: Custom 4D multi-head attention for tabular data
   - Projects numeric data into higher dimensions via base numeric embeddings

2. **Training Infrastructure** (`hephaestus/training/`):
   - `training.py`: Contains `TabularDecoder`, a PyTorch Lightning module that wraps the TimeSeriesDecoder
   - Handles both time series and single-row prediction modes

3. **Data Handling** (`hephaestus/utils/`):
   - `numeric_categorical.py`: Manages mixed numeric/categorical data with NaN handling
   - `tokenizer.py`: Tokenization for categorical features
   - Automatic handling of missing values in both numeric and categorical columns

4. **Single Row Models** (`hephaestus/single_row_models/`):
   - Alternative implementation for non-time series tabular data
   - Currently being re-implemented according to roadmap

## Key Technical Details

- **Attention Mechanism**: Applies 4D multi-head attention where dimensions represent (batch, sequence, features, embedding)
- **Numeric Embeddings**: Numeric values are projected to higher dimensions by multiplying against learnable base embeddings
- **Output Constraints**: Can restrict outputs to valid categorical values
- **Multi-Output**: Designed to predict multiple targets simultaneously
- **PyTorch Lightning**: Uses Lightning for training loop management and experiment tracking

## Common Development Tasks

When modifying the attention mechanism, key files are:
- `hephaestus/timeseries_models/multihead_attention.py`
- `hephaestus/timeseries_models/models.py`

When working with data handling:
- `hephaestus/utils/numeric_categorical.py`
- `hephaestus/timeseries_models/encoder_decoder_dataset.py`

## Current Development Status

Active development on the anomaly-demo branch. The roadmap includes:
- Encoder-only mode implementation
- Re-implementing single row mode
- Re-implementing fine-tuning capabilities