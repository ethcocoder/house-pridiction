import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from src.core.base import ModelInterface
from src.data.utils import setup_logger
from src.config import PathConfig

logger = setup_logger("BaselineModel", log_file=os.path.join(PathConfig.LOGS_DIR, "modeling.log") if os.path.exists(PathConfig.LOGS_DIR) else "modeling.log")

class BaselineModel(ModelInterface):
    """Linear Regression Baseline Model."""
    
    def __init__(self):
        self.model = None
        
    def _build_pipeline(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Training Linear Regression baseline model...")
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        self.model = self._build_pipeline(numeric_features, categorical_features)
        self.model.fit(X, y)
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
        logger.info(f"Baseline pipeline saved to {path}")
        
        # Save preprocessor separately (preprocessing.pkl) as per documentation
        preprocessor = self.model.named_steps['preprocessor']
        prep_path = os.path.join(os.path.dirname(path), "preprocessing.pkl")
        joblib.dump(preprocessor, prep_path)
        logger.info(f"Baseline preprocessor saved to {prep_path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
