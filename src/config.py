from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass(frozen=True)
class PathConfig:
    ROOT_DIR: str = "d:/house"
    DATA_DIR: str = os.path.join(ROOT_DIR, "data")
    RAW_DATA: str = os.path.join(DATA_DIR, "raw", "massive_housing_data.csv")
    PROCESSED_DATA: str = os.path.join(DATA_DIR, "processed", "housing_cleaned.csv")
    MODELS_DIR: str = os.path.join(ROOT_DIR, "models")
    LOGS_DIR: str = os.path.join(ROOT_DIR, "logs")
    REPORTS_DIR: str = os.path.join(ROOT_DIR, "reports")

@dataclass
class ModelConfig:
    TARGET_COL: str = "SalePrice"
    LOG_TRANSFORM: bool = True
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # XGBoost Params
    XGB_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1
    })

