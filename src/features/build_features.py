import pandas as pd
import os
from src.core.base import DataProcessor
from src.data.utils import setup_logger
from src.config import PathConfig

logger = setup_logger("FeatureEngineering", log_file=os.path.join(PathConfig.LOGS_DIR, "features.log") if os.path.exists(PathConfig.LOGS_DIR) else "features.log")

class FeatureEngineerProcessor(DataProcessor):
    """Engineers new features for the housing dataset."""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Starting FeatureEngineerProcessor...")
        
        df = self._add_total_sf(df)
        df = self._add_age_features(df)
        df = self._add_total_bath(df)
        
        logger.info("Finished FeatureEngineerProcessor.")
        return df

    def _add_total_sf(self, df: pd.DataFrame) -> pd.DataFrame:
        req_cols = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
        if all(col in df.columns for col in req_cols):
            df['TotalSF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
        return df

    def _add_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(col in df.columns for col in ['Yr Sold', 'Year Built']):
            df['HouseAge'] = df['Yr Sold'] - df['Year Built']
        
        if all(col in df.columns for col in ['Yr Sold', 'Year Remod/Add']):
            df['YearsSinceRemodel'] = df['Yr Sold'] - df['Year Remod/Add']
        return df

    def _add_total_bath(self, df: pd.DataFrame) -> pd.DataFrame:
        bath_cols = ['Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath']
        if all(col in df.columns for col in bath_cols):
            df['TotalBath'] = df['Full Bath'] + (0.5 * df['Half Bath']) + \
                               df['Bsmt Full Bath'] + (0.5 * df['Bsmt Half Bath'])
        return df
