# Portfolio vs S&P500 — Outperformance Prediction (2013–2023)

## 1. Research question

This project predicts whether an equally weighted portfolio of AAPL, AMZN, and MSFT will outperform the S&P 500 over the next 20 days, using machine-learning models and historical financial data.


## 2. Repository structure

The repository is organized as follows:

- `notebooks/` – Interactive analysis and visualizations  
  - `data_exploration.ipynb` – Main notebook: data loading, plots, feature exploration  
  - `TheDataProject_Notebook.ipynb` – Additional notebook from earlier stages of the project  

- `src/` – Core project code  
  - `data_loader.py` – Downloading data, preprocessing, feature engineering, label creation  
  - `models.py` – Model definitions and training utilities  
  - `evaluation.py` – Evaluation metrics and plotting functions  

- `main.py` – End-to-end pipeline script (runs the whole project from the command line)  
- `environment.yml` – Conda environment with all dependencies  
- `README.md` – Project documentation  
- `.gitignore` – Files and folders ignored by Git


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

### Expected output

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

No files are saved to disk. Results are printed in the terminal and visualized in notebooks.



## 5. Expected outputs (what you should obtain)

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
- Test Accuracy: 0.89
- Precision: 0.84 / 0.94
- Recall: 0.91 / 0.89
- F1-score: 0.88 / 0.92

**Interpretation (short):**
- Logistic Regression performs consistently and does not overfit.
- Random Forest performs better overall and remains stable, suggesting good generalization.




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

## AI Usage

AI tools were used as support only. I stayed responsible for every decision, explanation, and line of code.

### Tools
- ChatGPT Pro (since October 2023)
- OpenAI Codex (used occasionally inside VS Code)

### What AI helped with
- Clarifying concepts (models, metrics, workflows)
- Explaining errors and debugging strategies
- Generating small boilerplate code (function templates, plotting skeletons)
- Improving documentation structure and wording
- Suggestions for refactoring and tests

### How I used and verified AI
- I reviewed every suggestion carefully
- I rewrote code to fit my own design
- I tested everything manually
- I removed anything unclear or incorrect

AI did not write the project for me.  
It acted as a tutor and assistant.

### Reflection
Using AI required me to think like an “information architect”:
asking better questions, selecting useful ideas, and verifying everything before using it.



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