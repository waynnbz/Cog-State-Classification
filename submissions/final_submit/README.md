# Running Instructions
- please refer to the README.md within `/code` folder for default running procedures. `-v` for external data is expected.

# Feature Engineering

## Missing value handling
- all sensordata columns are first convert to numeric values and then filled with -9999.9 to eliminating outliers/missing values
- `-9999.9` is then auto handled by the XGBoost model

## Data Compose
- data within 1s timeframe are combined via taking a mean(exluding default values)
- data are loading by chunks for RAM concerns, duplicates between diffferent chunks have further aggregated

# Model Selection
- mainly applied XGBoost tree model for the multi-class classification
- model parameters configuration can be found in the `src/sample/submission/config.py` file
- T-3s is found always stay the same for given training data, so same prediction model is used
- Overall Top-Three-Feature is observed and written to the solution file

# Possible future work
- could have investigate the model performance on full training data without merging them, although the merge is able to fillup missing info within the second if it's present eleswhere.