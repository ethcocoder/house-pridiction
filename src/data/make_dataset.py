import pandas as pd
import os
import argparse
from src.data.cleaning import CleaningProcessor, Pipeline
from src.features.build_features import FeatureEngineerProcessor
from src.config import PathConfig
from src.data.utils import setup_logger

logger = setup_logger("MakeDataset", log_file=os.path.join(PathConfig.LOGS_DIR, "make_dataset.log") if os.path.exists(PathConfig.LOGS_DIR) else "make_dataset.log")

def main():
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline.")
    parser.add_argument("--input", type=str, default=PathConfig.RAW_DATA, help="Path to raw data")
    parser.add_argument("--output", type=str, default=PathConfig.PROCESSED_DATA, help="Path to save processed data")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Loading raw data from {args.input}")
    df = pd.read_csv(args.input)

    # Initialize professional pipeline
    pipeline = Pipeline([
        CleaningProcessor(missing_threshold=0.8),
        FeatureEngineerProcessor()
    ])

    # Execute pipeline
    processed_df = pipeline.run(df)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save
    processed_df.to_csv(args.output, index=False)
    logger.info(f"Successfully saved processed data to {args.output}")

if __name__ == "__main__":
    main()
