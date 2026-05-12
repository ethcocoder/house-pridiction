from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class DataProcessor(ABC):
    """Abstract base class for all data processing steps."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the input dataframe and return the transformed version."""
        pass

class ModelInterface(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize the model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Deserialize the model from disk."""
        pass
