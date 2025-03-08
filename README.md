# Hephaestus

Hephaestus is a Python package/neural network designed to apply the attention to tabular
data.

It's key features are:

-   Natively handling numbers and categorical data.
-   Applying attention to tabular data.
-   Handling missing data.
-   Predicting multiple outputs at the same time (a strength and weakness).
-   Limiting the outputs to a set of possible values (a strength and weakness).

## Underlying Principles

Hephaestus projects numeric data into a higher-dimensional space via multiplication of
the numeric values against base numeric embedding.

Results are restricted to the number of columns of categorical and numeric data.

## Roadmap

-   [ ] Add Encoder only mode.
-   [ ] Add Encoder-Decoder mode.
-   [ ] Add single row (non-timeseries) mode (previously implemented but removed).
-   [ ] Add fine tuning mode (previously implemented but removed).

## Install

```bash
uv pip install -e .
```
