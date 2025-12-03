# Portfolio vs S&P500: Outperformance Prediction (2013–2023)

## Research Question

Can we predict whether an **equally weighted portfolio of AAPL, AMZN and MSFT** will outperform the **S&P 500** over the next 20 trading days, using only a few risk/return features (volatility, Sharpe ratio, covariance with the market)?

More formally, the target variable is:

- `result = 1` if the portfolio return over the next 20 days is **higher** than the S&P500 return,
- `result = 0` otherwise.

---

## Data

The data are downloaded directly from Yahoo Finance with the `yfinance` library:

- **Assets in the portfolio**
  - Apple (`AAPL`)
  - Amazon (`AMZN`)
  - Microsoft (`MSFT`)
- **Market benchmark**
  - S&P 500 index (`^GSPC`)
- **Risk-free rate proxy**
  - 10-Year US Treasury Yield (`^TNX`)

**Period:** from `2013-01-01` to `2023-12-31`  
**Frequency:** daily prices (1d interval)

No external files are required: everything is pulled online via `yfinance`.

---

## Feature Engineering

Starting from daily closing prices, the notebook builds several intermediate objects and finally a modelling dataset `df_final`.

### 1. Daily returns

For each asset and for the S&P500:

r_t = (P_t / P_{t-1}) - 1

An **equally weighted portfolio** of AAPL, AMZN and MSFT is created:

- Combine the three series of daily returns
- Take the mean across the three assets → `rendement_portefeuille["moyenne"]`

### 2. Normalised prices

For some plots and volatility calculations, prices are normalised by their first value:

P_norm(t) = P_t / P_0

This is done both for the portfolio and for the S&P500.

### 3. Rolling risk measures (20-day window)

On **20-day rolling windows**, we compute:

- **Rolling variance** of the portfolio and of the S&P500
- **Rolling mean** of prices  
- **Coefficient of variation (CV)** as a proxy for volatility:

CV = variance / mean

This gives:
- `cv_rolling_port` for the portfolio  
- `cv_rolling_sp` for the S&P500

### 4. Risk-free rate and Sharpe ratio

The 10-year Treasury yield (`^TNX`) is downloaded and converted to an approximate **daily risk-free rate**:

- 20-day rolling mean of the yield
- Divided by 250 (approx. trading days per year) → `rf_tout`

On the same 20-day windows, we compute **block returns**:

Block return = (P_last − P_first) / P_first

- `r_port` and `r_port_mean`: 20-day portfolio returns
- `r_sp500`: 20-day S&P500 returns

Then the **Sharpe ratios** are approximated as follows:

 Portfolio Sharpe ratio:  
  (Portfolio return − risk-free rate) / rolling coefficient of variation  

 S&P500 Sharpe ratio:  
  (S&P500 return − risk-free rate) / rolling coefficient of variation

### 5. Covariance with the market

Using daily returns, a **20-day rolling covariance** between the portfolio and the S&P500 is computed:

- Combine portfolio daily returns and S&P500 daily returns in a DataFrame
- Use `.rolling(window=20).cov()` to obtain `cov_by_block`

### 6. Target variable (outperformance)

To define the classification label, we look at **21-day windows**:

- 21-day block returns for the portfolio: `r_port_21`
- 21-day block returns for the S&P500: `r_sp500_21`

Then:

```python
df_final["result"] = (r_port_mean_21 > r_sp500_21).astype(int)



## Note: Feature scaling is applied after the train/test split. 
The scaler is fitted on the training data only to prevent data leakage.



## Notes on Model Evaluation and Debugging

Two different executions of the project are provided:

- The Jupyter notebook (`TheDataProject_Notebook.ipynb`) is designed for exploratory analysis and experimentation on a larger, more realistic dataset. In this setting, the model reaches an accuracy of approximately **0.80**, which reflects a more challenging and realistic classification task.

- The command-line pipeline (`python main.py`) is a fully reproducible, modular implementation using a clean dataset and a fixed train/test split. In this configuration, the models achieve very high performance (up to 1.00 accuracy on the test set), which is expected for this dataset and mainly serves to validate the correctness of the pipeline, the preprocessing steps, and the model implementations.

This difference in performance is intentional and illustrates an important machine learning principle:  
**high accuracy can depend strongly on dataset simplicity and experimental setup**, and should always be interpreted in context.

The project structure (separate data loading, preprocessing, training, and evaluation steps) also makes it easy to debug common machine learning issues such as shape mismatches, incorrect data types, or missing values, as highlighted in the course material.

Regarding overfitting, model performance is monitored by comparing training and test accuracy. In the exploratory notebook, performance remains moderate and consistent across splits (≈0.80), suggesting limited overfitting. In the command-line pipeline, very high accuracy is observed due to the simplicity of the dataset, which is a known and expected behavior rather than an overfitting issue.



## Note on Reported Results

Two different execution contexts are provided in this project:

- The command-line pipeline (`main.py`), executed on the full benchmark dataset,  
  where some models may reach very high accuracy due to the simplicity 
  and clear separability of the Iris dataset.

- The exploratory notebook (`TheDataProject_Notebook.ipynb`), where additional preprocessing,
  different train/test splits and intermediate checks are performed.  
  In this setting, the observed accuracy is around 0.80, which is more representative of a
  realistic evaluation scenario.

This discrepancy is expected and intentional:
the notebook is used for experimentation and validation,
while `main.py` serves as a clean, reproducible entry point to demonstrate the full ML pipeline.

The goal of this project is not to maximize accuracy, but to illustrate good software engineering
and machine learning practices.



## Reproducibility and Randomness

Some machine learning algorithms, such as Random Forests, rely on randomness (e.g. bootstrapping samples or selecting random feature subsets).  
As a result, running the same code multiple times can yield slightly different performance metrics.

In this project, reproducibility is ensured in the command-line pipeline (`main.py`) by fixing random seeds where applicable. This guarantees stable and repeatable results when the project is executed from the repository.

In contrast, the Jupyter notebook intentionally illustrates more exploratory behavior, where performance variations may occur across runs. This distinction reflects common practices in machine learning between reproducible pipelines and experimental analysis.

Where applicable, random seeds (`random_state`) are explicitly fixed (e.g. in train/test splitting and tree-based models) to ensure fully reproducible results when running the command-line pipeline. The same random seed is used consistently across the pipeline, following best practices for reproducible machine learning experiments.



## Results and Notes on Model Performance

The command-line script (`main.py`) uses the Iris dataset to demonstrate a clean, fully reproducible machine learning pipeline.  
Due to the simplicity and small size of the Iris dataset, several models (Random Forest, KNN, Logistic Regression) achieve an accuracy of 1.00 on the test set. This result is expected and commonly observed in educational examples.

The accompanying Jupyter notebook (`TheDataProject_Notebook.ipynb`) presents experiments on a more realistic dataset with greater variability and noise. In this setting, model performance is lower, with an average accuracy of approximately **0.80**, which better reflects real-world machine learning conditions.

This difference in performance highlights an important concept:
- High accuracy on small, clean datasets does not necessarily generalize to real-world data.
- Model evaluation depends strongly on data complexity, preprocessing choices, and experimental setup.

Both implementations are intentionally included:
- `main.py` focuses on **reproducibility, structure, and best practices.
- The notebook focuses on **exploration, realism, and practical limitations of machine learning models.
