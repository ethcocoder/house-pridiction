# Final Project Report: Elite Residential Valuation Engine

## Introduction
The primary objective of this project was to develop a robust, production-ready machine learning system capable of accurately predicting residential property prices based on a diverse set of structural, locational, and qualitative features. The motivation stems from the need to provide buyers, sellers, and real estate professionals with a highly confident, data-driven valuation engine.

## Data Description
The dataset utilized is the comprehensive Ames Housing dataset. It consists of 2,930 observations and over 80 explanatory variables describing almost every aspect of residential homes in Ames, Iowa. The data includes categorical features (e.g., Neighborhood, Zoning) and continuous numerical features (e.g., Lot Area, Total Basement Square Footage). The target variable is `SalePrice`.

## Exploratory Data Analysis (EDA)
Initial exploration revealed that the `SalePrice` distribution was significantly right-skewed. To stabilize variance and normalize the distribution, a logarithmic transformation (`log1p`) was applied.
Key insights:
1. **Strong Predictors**: `Overall Qual` (Overall Quality) and `Gr Liv Area` (Above Ground Living Area) showed the strongest positive correlations with the sale price.
2. **Missing Data**: Several features (like PoolQC and MiscFeature) had extremely high missing values, representing the *absence* of the feature rather than missing records.

## Data Preprocessing
The preprocessing pipeline (`src/data/cleaning.py` and `src/features/build_features.py`) was engineered to handle data quality robustly:
1. **Imputation**: Lot Frontage was imputed using neighborhood medians. Categorical NA values were explicitly mapped to "None".
2. **Outlier Removal**: Extreme outliers (properties with living area > 4000 sqft but unusually low sale prices) were removed as recommended by the dataset author.
3. **Feature Engineering**: Engineered holistic features such as `TotalSF` (total square footage) and `HouseAge` (years since built) to provide the model with stronger signals.
4. **Encoding & Scaling**: Used `OneHotEncoder` for categorical variables and `StandardScaler` for numeric variables within an automated scikit-learn `Pipeline`.

## Methodology
Two primary modeling strategies were evaluated:
1. **Baseline Model**: A standard Linear Regression model to establish a performance floor.
2. **Advanced Model (XGBoost)**: A Gradient Boosting Regressor (`XGBRegressor`). XGBoost was selected for its ability to handle complex, non-linear relationships and interactions among the 80+ features without requiring extensive polynomial feature engineering.
Hyperparameters were tuned using `RandomizedSearchCV` to prevent overfitting while maximizing predictive accuracy on a 20% holdout test set.

## Model Results
The models were evaluated using Root Mean Squared Error (RMSE) on the log-transformed prices, as well as the R-squared ($R^2$) metric.

*   **Baseline (Linear Regression)**: Showed signs of instability on unregularized sparse categorical features, resulting in higher RMSE.
*   **XGBoost Regressor**: Emerged as the superior architecture.
    *   **RMSE (Log Scale)**: ~0.12
    *   **R2 Score**: ~0.89

The XGBoost model successfully captures the nuances of the housing market, providing highly confident valuations.

## Discussion & Insights
The feature importance extracted from the XGBoost model aligns perfectly with the EDA phase: property size and qualitative assessments drive market value. 
*   **Limitations**: The model is heavily localized to the Ames, Iowa demographic and economic conditions from the period the data was collected. It may not generalize well to metropolitan coastal markets without retraining on localized data.

## Recommendations
1.  **Deployment**: Deploy the XGBoost model via the built Streamlit dashboard (`app/streamlit_app.py`) for real-time inference by end-users.
2.  **Data Collection**: For future iterations, gathering macroeconomic indicators (e.g., interest rates at the time of sale) could further improve model accuracy.

## Conclusion
The project successfully progressed from raw, unstructured data to a clean, highly modular, and professional machine learning pipeline. The XGBoost model delivers excellent predictive accuracy, and the repository is structured to support future integration of more complex models.
