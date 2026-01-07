## AI Usage



### Tools Used
- ChatGPT Pro (since October 2023)
- OpenAI Codex (used occasionally inside VS Code)



## Significant Contributions

1. **Writing and Communication**
  AI also helped me refine my English writing: improving clarity, grammar, and style.

2. **Project Orchestration** (`main.py`)
- **Academic Basis:** Follows the modular workflow and software efficiency principles taught in Week 6 (Software Engineering) and the classification pipelines from Week 8.
- **AI Usage:** Assisted in automating results folder management (`os.makedirs`) and structuring the CLI entry point.

3. Data Loading & Engineering (`src/data_loader.py`)
- **Academic Basis:** Built upon Weeks 3 & 4 (Python Fundamentals) and Week 7 (Linear Regression) for financial data preprocessing. 
- **AI Usage:** Helpful for managing complex `yfinance` multi-indexes and optimizing data normalization functions.
- **Human Ownership:** I designed the feature engineering logic, specifically the 20-day smoothed risk-free rate and Sharpe ratio calculations.

4. Model Definitions (`src/models.py`)
- **Academic Basis:** Implements **Logistic Regression (Week 7)** and **Random Forest (Week 8)**.
- **AI Usage:** Assisted in implementing flexible `**kwargs` for hyperparameter tuning.
- **Human Ownership:** I chose to compare linear vs. non-linear models to analyze financial risk, a strategy learned in Week 11 (Advanced ML).

5. Evaluation & Metrics (`src/evaluation.py`)
- **Academic Basis:** Direct application of Week 8 (Classification) metrics: Confusion Matrix, Precision, Recall, and ROC-AUC.
- **AI Usage:** Supported the drafting of professional docstrings and formatting complex comparison plots.

6. Exploratory Analysis (`notebooks/`)
- **Academic Basis:** Uses EDA techniques and covariance analysis related to Week 9 (Unsupervised Learning) and Week 7.
- **AI Usage:** Acted as a technical guide for advanced `matplotlib` formatting and time-series visualization.

7. Validation & Testing (`tests/`)
- **Academic Basis:** Implements Unit Testing principles from Week 6 (Software Engineering).
- **AI Usage:** Suggested edge cases for data validation and assisted with `pytest` configuration.



## Learning Moments
- I learned how to structure a full ML workflow (data → features → models → evaluation)
- I understood better the difference between accuracy, precision, recall and ROC–AUC
- I improved my debugging process by asking more precise questions and testing step by step
- I realized that AI is useful as a guide, but the final design and decisions must come from me
- I became more careful about verifying every suggestion instead of copy-pasting



### Reflection
Before working on this project, I had already spent time reading about AI use in learning,
including Laurent Alexandre’s book *La guerre des intelligences à l’heure de GPT*
and the MIT Media Lab study “Your Brain on ChatGPT”. 

The MIT Media Lab study shows that AI can improve speed and accuracy,
but it also reduces active reasoning: instead of solving problems, the brain shifts toward
checking and approving AI answers. This can create an illusion of competence over time.
Keeping this in mind, I used AI carefully, as support for understanding, not as a substitute
for thinking or learning.

These works highlight both the opportunities and the cognitive risks of relying too much on AI.

With that in mind, I was intentional about how I used AI in this project.
I focused on understanding concepts first, keeping control over design choices,
and using AI mainly to clarify ideas when I was blocked.

Rather than replacing my work, AI acted as a complementary tool, useful for guidance,
but never a substitute for thinking, practice, and ownership of the final result.