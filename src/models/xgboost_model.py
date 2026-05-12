import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from src.core.base import ModelInterface
from src.data.utils import setup_logger
from src.config import PathConfig, ModelConfig

logger = setup_logger("XGBoostModel", log_file=os.path.join(PathConfig.LOGS_DIR, "modeling.log") if os.path.exists(PathConfig.LOGS_DIR) else "modeling.log")

class XGBoostModel(ModelInterface):
    """Advanced XGBoost Model with Hyperparameter Tuning."""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def _build_pipeline(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        xgb = XGBRegressor(objective='reg:squarederror', random_state=ModelConfig.RANDOM_STATE)
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Starting Hyperparameter Tuning for XGBoost...")
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        pipeline = self._build_pipeline(numeric_features, categorical_features)
        
        # Hyperparameter Grid
        param_dist = {
            'regressor__n_estimators': [100, 500, 1000],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__subsample': [0.7, 0.8, 0.9],
            'regressor__colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=5, 
            cv=3, scoring='neg_mean_squared_error', verbose=2, 
            random_state=ModelConfig.RANDOM_STATE, n_jobs=1
        )
        
        random_search.fit(X, y)
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        logger.info(f"Best Parameters: {self.best_params}")
        logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        
        # Save full pipeline (model.pkl)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model pipeline saved to {path}")
        
        # Save preprocessor separately (preprocessing.pkl) as per documentation
        preprocessor = self.model.named_steps['preprocessor']
        prep_path = os.path.join(os.path.dirname(path), "preprocessing.pkl")
        joblib.dump(preprocessor, prep_path)
        logger.info(f"Preprocessor saved to {prep_path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    # 1. Load Processed Data
    if not os.path.exists(PathConfig.PROCESSED_DATA):
        print(f"Error: Processed data not found at {PathConfig.PROCESSED_DATA}. Run 'python -m src.data.make_dataset' first.")
    else:
        df = pd.read_csv(PathConfig.PROCESSED_DATA)
        X = df.drop(columns=[ModelConfig.TARGET_COL])
        y = np.log1p(df[ModelConfig.TARGET_COL])
        
        # 2. Train and Save
        model = XGBoostModel()
        model.train(X, y)
        model.save(os.path.join(PathConfig.MODELS_DIR, "model.pkl"))

