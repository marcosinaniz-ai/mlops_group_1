# Business Case – Health Insurance Cost Prediction

**Author:** Eduardo Debes, Nicolas Guadamillas, Benedetta Pagliardi,
Marco Siñaniz, Max Vanderlinden
**Course:** MLOps: Master in Business Analytics and Data Sciense
**Status:** Session 1 (Initialization)

---

## 1. Business Objective
This project transforms our previous machine learning model for predicting individual health insurance charges into a production-ready MLOps pipeline. The objective is to support health insurance pricing decisions by estimating expected medical costs based on demographic and lifestyle attributes.
* **The Goal:** Develop a production-ready machine learning system that predicts individual health insurance charges based on demographic and lifestyle attributes.
* **The User:** The primary users are actuarial analysts and underwriting teams within a health insurance company. They use the model’s predicted cost estimates to support premium pricing decisions, risk assessment, and portfolio analysis, helping ensure that individual policies are aligned with expected healthcare expenses.

---

## 2. Success Metrics
*How do we know if the project is successful?*

* **Business KPI (The "Why"):** The project is successful if the model improves pricing accuracy by reducing the gap between predicted and actual healthcare costs. More precise predictions support better risk-based pricing decisions and contribute to more stable loss ratios over time.

* **Technical Metric (The "How"):** As this is a regression problem, performance is evaluated using R² and RMSE. The model should achieve an R² of at least 0.75 on the test set while maintaining stable RMSE across runs.

* **Acceptance Criteria:** The model must outperform a simple baseline (such as predicting the mean cost), execute reproducibly through the main pipeline, and pass all validation and automated tests.

---

## 3. The Data

* **Source:** The dataset is a public health insurance dataset originally used in our previous machine learning project. It contains individual-level demographic and lifestyle information along with associated medical charges.
* **Target Variable:** The target variable is charges, representing the total healthcare cost incurred by an individual. This is treated as a regression problem.
* **Sensitive Info:** The dataset is publicly available and anonymized. It does not contain personally identifiable information. All data directories are excluded from version control using .gitignore as a best practice.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── 01_health_insurance_baseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── utils.py             # Centralize simple I/O primitives
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   ├── features.py          # Feature preprocessing
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Testing all the src python scripts
│   ├── __init__.py          # Python package
│   ├── test_load_data.py        
│   ├── test_clean_data.py       
│   ├── test_validate.py          
│   ├── test_train.py
│   ├── test_utils.py               
│   ├── test_evaluate.py
│   ├── test_infer.py
│   ├── test_features.py     
│   └── test_main.py             
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python src/main.py`



