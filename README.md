# Portfolio vs S&P500 — Outperformance Prediction (2013–2023)


## 1. Research question

This project predicts whether an equally weighted portfolio of AAPL, AMZN, and MSFT will outperform the S&P 500 over the next 20 days, using machine-learning models and historical financial data.


## Project Report

The full project report is available as a PDF:

- `project_report.pdf`

For optimal reading, click directly on the file from the GitHub repository to open it in the browser or download it locally.


## 2. Repository structure

The repository is organized as follows:

- `figures/` – Saved evaluation plots  
  - ROC curve, precision–recall curve, confusion matrix, probability distributions  

- `notebooks/` – Interactive analysis and visualizations  
  - `data_exploration_Final.ipynb` – Main notebook: data loading, plots, feature exploration  

- `src/` – Core project code  
  - `data_loader.py` – Downloading data, preprocessing, feature engineering, label creation  
  - `models.py` – Model definitions and training utilities  
  - `evaluation.py` – Evaluation metrics and plotting functions  

- `tests/` – Unit, integration, performance, error-handling tests  

- `.gitignore` – Files and folders ignored by Git
- `AI_USAGE.md` – Description of AI assistance used in the project 
- `environment.yml` – Conda environment with all dependencies
- `main.py` – End-to-end pipeline script (runs the whole project from the command line)  
- `project_report.pdf` – Final project report (PDF)  
- `project_report.tex` – LaTeX source of the project report 
- `pytest.ini` – Pytest configuration file  
- `README.md` – Project documentation  


## 3. Installation

This project uses a Conda environment to manage all dependencies.

From the root of the repository:

```bash
conda env create -f environment.yml
conda activate final-data-project
```


## 4. How to run the project

Run the full pipeline from the root of the project:

```bash
python main.py
```

The project downloads financial data from Yahoo Finance at runtime.
If the connection fails (rate limit, no internet, or API error), the script may stop and the results/ folder may not be created.
Simply re-run the script after a few minutes.


### Expected console output

When running:

```bash
python main.py
```

you should see messages similar to:

```
Loading data...
Training models...
Evaluating models...

Accuracy, precision, recall and F1-score printed for each model.
```

The evaluation plots are saved for each model in the `results/` folder. 


## 5. Outputs (what you should obtain)

After running the project, you should obtain:

###  Plots
From the notebook:
- Rolling returns and benchmark comparison
- Rolling Sharpe ratios
- Rolling outperformance fraction
- Distribution plots of risk / performance metrics

These visualizations help understand how the portfolio behaves vs the S&P500 over time.


###  Metrics printed in terminal (when running `python main.py`)

For each model, you should see evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report


###  Internal project artifacts (not committed)

During execution, the project temporarily creates in-memory objects such as:

- cleaned datasets
- feature matrices
- trained model objects
- prediction labels

These are used only for computation and are **not saved as files**, unless added later.

In short: if you see the plots in the notebook and the metrics printed in the terminal, the pipeline works correctly.


### 5.1 Model results (metrics)

Below are the main evaluation results obtained when running `python main.py`.

#### Model: Logistic Regression
- Test Accuracy: 0.80
- Precision: 0.84 (class 0) / 0.78 (class 1)
- Recall: 0.59 (class 0) / 0.93 (class 1)
- F1-score: 0.69 / 0.85

#### Model: Random Forest
- Test Accuracy: 0.91
- Precision: 0.86 / 0.95
- Recall: 0.92 / 0.90
- F1-score: 0.89 / 0.93

**Interpretation (short):**
- Logistic Regression performs consistently and does not overfit.
- Random Forest performs better overall and remains stable, suggesting good generalization.

### ROC–AUC

To better evaluate classifier performance, we also computed the ROC–AUC score.

The ROC curve compares the True Positive Rate vs. False Positive Rate across
different thresholds.

Result:

- **AUC = 0.96**

An AUC close to 1.0 indicates strong discriminative ability.  
A value of 0.89 suggests the model distinguishes well between
“outperform” vs “not outperform”, without relying only on accuracy.


## 6. Data

The project uses publicly available financial market data:

- **Stock prices** (AAPL, AMZN, MSFT)
- **S&P 500 benchmark**
- **Risk-free rate (TNX – 10-year Treasury)**

The data are downloaded automatically from Yahoo Finance via the Python library `yfinance`.

You do NOT need to download anything manually.

All downloads happen inside:

`src/data_loader.py`

If the user wants to change tickers or dates, they can modify the parameters directly in that file.


## 7. Requirements

- Python 3.11
- Conda (to create the `final-data-project` environment)
- Main Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - yfinance


## 8. Test suite — Results & Coverage

We implemented a complete testing pipeline covering different aspects of the project.

####  Unit tests (core functions)
- `test_data_loader.py`
  Verifies that financial data loads correctly and returns valid DataFrames.

- `test_models.py`
  Ensures Logistic Regression and Random Forest:
  - train without crashing
  - make predictions
  - achieve accuracy > 0.5 (better than random)

####  Integration test
- `test_integration_main.py`
  Runs the full pipeline (`python main.py`) and checks that execution finishes successfully.

####  Error-handling test
- `test_error_handling.py`
  Confirms invalid input shapes raise appropriate exceptions.

####  Performance test
- `test_performance.py`
  Ensures model training remains within a reasonable time budget.


### Test coverage

Coverage was computed using:

```bash
pytest --cov=src --cov-report=term-missing
```

**TOTAL: 88% coverage**

I am aware that recommended coverage is above 70%.
At this stage, the project clearly exceeds that target.

Most of the remaining uncovered lines belong to plotting utilities and exploratory helpers,
which are not critical for correctness. The essential pipeline
(data loading, model training, evaluation, integration, and error handling)
is now well covered.