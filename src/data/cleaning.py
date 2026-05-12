import pandas as pd
import numpy as np
import os
from typing import List
from src.core.base import DataProcessor
from src.data.utils import setup_logger
from src.config import PathConfig

logger = setup_logger("CleaningProcessor", log_file=os.path.join(PathConfig.LOGS_DIR, "preprocessing.log") if os.path.exists(PathConfig.LOGS_DIR) else "preprocessing.log")

class CleaningProcessor(DataProcessor):
    """Handles missing values, outliers, and basic cleaning."""
    
    def __init__(self, missing_threshold: float = 0.8):
        self.missing_threshold = missing_threshold

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Starting CleaningProcessor on dataframe of shape {df.shape}")
        
        df = self._drop_high_missing_cols(df)
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        
        logger.info(f"Finished CleaningProcessor. Output shape: {df.shape}")
        return df

    def _drop_high_missing_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        logger.info(f"Dropping columns with >{self.missing_threshold*100}% missing: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values...")
        
        if 'Lot Frontage' in df.columns and 'Neighborhood' in df.columns:
            df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(
                lambda x: x.fillna(x.median())
            )
            df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())

        cat_none_cols = [
            'Alley', 'Mas Vnr Type', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 
            'BsmtFin Type 1', 'BsmtFin Type 2', 'Fireplace Qu', 'Garage Type', 
            'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence', 'Misc Feature'
        ]
        for col in cat_none_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        num_zero_cols = [
            'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 
            'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Yr Blt', 
            'Garage Cars', 'Garage Area'
        ]
        for col in num_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if 'Electrical' in df.columns:
            df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing recommended outliers...")
        if 'Gr Liv Area' in df.columns and 'SalePrice' in df.columns:
            initial_count = len(df)
            df = df[~((df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 300000))]
            dropped = initial_count - len(df)
            logger.info(f"Dropped {dropped} outliers.")
        return df

class Pipeline:
    """Executes a series of DataProcessors."""
    def __init__(self, processors: List[DataProcessor]):
        self.processors = processors
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for processor in self.processors:
            result = processor.process(result)
        return result
