# Elite Residential Valuation Engine

## Project Overview
This project is a comprehensive data science application designed to predict house prices using advanced machine learning techniques. It covers the entire lifecycle from exploratory data analysis (EDA) and data preprocessing to model training, evaluation, and deployment via an interactive Streamlit application. The core intelligence engine utilizes an optimized XGBoost model to provide highly accurate, log-transformed price estimations based on a robust set of property characteristics.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation Instructions](#installation-instructions)
3. [How to Run the Code](#how-to-run-the-code)
4. [Project Structure](#project-structure)
5. [Key Findings and Insights](#key-findings-and-insights)
6. [Final Report](#final-report)

## Installation Instructions
1. Clone the repository to your local machine.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Run the Code

### 1. Data Pipeline
To process the raw data and generate the engineered dataset:
```bash
python -m src.data.make_dataset
```

### 2. Model Training
To train the baseline and advanced XGBoost models:
```bash
# Run the notebooks sequentially for detailed walkthroughs
jupyter notebook notebooks/
```
Or execute the core logic directly (requires adjusting main blocks if run as scripts):
```bash
python -m src.models.baseline
python -m src.models.xgboost_model
```

### 3. Application Deployment
To launch the interactive prediction dashboard:
```bash
streamlit run app/streamlit_app.py
```

## Project Structure
```text
project-repository/
├── data/                  # Data directory
│   ├── raw/               # Original, immutable data
│   ├── processed/         # Cleaned, transformed data
│   └── external/          # Third-party data sources
├── notebooks/             # Jupyter notebooks for iterative analysis
│   ├── 01_eda.ipynb 
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb 
│   └── 04_evaluation.ipynb
├── src/                   # Source code
│   ├── data/              # Data processing & utilities
│   ├── features/          # Feature engineering
│   ├── models/            # Model architectures and pipelines
│   └── visualization/     # Plotting utilities
├── models/                # Saved serialized models and metadata
├── reports/               # Generated reports and figures
├── app/                   # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Key Findings and Insights
* **Overall Quality and Living Area**: These two features exhibit the strongest positive correlation with the final sale price, acting as primary drivers for the model's predictions.
* **Non-Linear Relationships**: The XGBoost engine significantly outperforms the baseline Linear Regression model by capturing complex, non-linear interactions between features such as neighborhood pricing dynamics and age/remodel combinations.
* **Log Transformation**: Applying a logarithmic transformation to the target variable (`SalePrice`) was crucial for normalizing the right-skewed distribution, resulting in more stable and robust error metrics (RMSE).

## Final Report
For an in-depth technical walkthrough of the methodology, evaluation metrics, and strategic recommendations, please refer to the [Final Report](reports/final_report.md).
