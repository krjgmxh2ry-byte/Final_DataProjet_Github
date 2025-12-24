...existing code...
# Portfolio vs S&P500 — Outperformance Prediction (2013–2023)

## Research question
Which classification model best predicts whether an equally-weighted portfolio [AAPL, AMZN, MSFT] will outperform the S&P500 over the next 20 days?  
(Compare Random Forest, Logistic Regression, etc.)

## Setup
1. Create the conda environment and activate it:
   ```sh
   conda env create -f environment.yml
   conda activate final-data-project
   ```
2. In VS Code: open the project folder and select the Python interpreter from the conda environment.  
   For Jupyter notebooks, install/select the environment kernel if needed.

## Usage
- Run the full pipeline (use the integrated terminal in VS Code):
  ```sh
  python main.py
  ```
- Interactive exploration:
  - Open `notebooks/data_exploration.ipynb` in VS Code or Jupyter.
  - Choose the `final-data-project` kernel.

## Project summary
- Goal: predict whether the equally-weighted portfolio [AAPL, AMZN, MSFT] will outperform the S&P500 over a 20-day horizon.
- Data: downloaded via `yfinance` (see `src/data_loader.py`).

## Project structure
- `main.py` — main pipeline script  
- `src/`
  - `src/data_loader.py` — data download, preprocessing, feature engineering, label creation
  - `src/models.py` — model training functions
  - `src/evaluation.py` — evaluation metrics and plotting utilities
- `notebooks/data_exploration.ipynb` — interactive exploration  
- `environment.yml` — environment and dependencies

## Pipeline (where to look in code)
- Data loading & split: `src.data_loader.load_and_split`  
- Core feature engineering: `src.data_loader.build_final_df_clean` (rolling CV, Sharpe, block returns, label)  
- Models: `src.models.train_logistic_regression`, `src.models.train_random_forest`  
- Evaluation & plots: functions in `src.evaluation`

## Feature engineering (key formulas)
- Daily return: r_t = P_t / P_{t-1} - 1  
- Block/window return: R_block = (P_last - P_first) / P_first  
- Coefficient of variation (CV): variance_rolling / mean_rolling  
- Approximate Sharpe: (R_window - rf) / CV

## Label definition
- Compare 21-day returns: r_port_21 vs r_sp500_21  
- `result` = 1 if r_port_21 > r_sp500_21, else 0  
  (implemented in `src.data_loader.build_final_df_clean`)

## Reproducibility
- `main.py` fixes random seeds for reproducibility.  
- For debugging, run notebook cells individually to inspect intermediate outputs.

## Results — PLACEHOLDER (fill with your final metrics)
- Model: Logistic Regression
  - Test Accuracy: ______
  - Precision: ______
  - Recall: ______
  - F1-score: ______
  - AUC: ______

- Model: Random Forest
  - Test Accuracy: ______
  - Precision: ______
  - Recall: ______
  - F1-score: ______
  - AUC: ______

- Notes on overfitting / stability:
  - ___________________________________________
  - ___________________________________________

## Tips & common pitfalls
- Scale features AFTER train/test split to avoid data leakage.  
- Ensure datetime indices are aligned (`.reindex(...)`) before rolling operations.  
- Useful functions to inspect:
  - `src.data_loader.compute_portfolio_returns`
  - `src.data_loader.compute_daily_return`
  - `src.evaluation.plot_roc_curve`

Files of interest
- `main.py`  
- `src/data_loader.py`  
- `src/models.py`  
- `src/evaluation.py`  
- `notebooks/data_exploration.ipynb`  
- `environment.yml`

License
- (Add license information if needed)

...existing code...