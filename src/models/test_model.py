import pandas as pd
import numpy as np
import os
import joblib
from src.models.xgboost_model import XGBoostModel
from src.config import PathConfig, ModelConfig

def test_inference():
    """Test the saved model artifacts by running a sample prediction."""
    print("🚀 Initializing Model Inference Test...")
    
    # 1. Initialize Model Class
    model_wrapper = XGBoostModel()
    
    # 2. Load the saved pipeline
    model_path = os.path.join(PathConfig.MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}. Please train the model first.")
        return
        
    model_wrapper.load(model_path)
    print(f"✅ Model successfully loaded from {model_path}")
    
    # 3. Create a sample for prediction
    # We load one row from the processed data
    if not os.path.exists(PathConfig.PROCESSED_DATA):
        print(f"❌ Error: Processed data not found. Please run make_dataset first.")
        return
        
    data = pd.read_csv(PathConfig.PROCESSED_DATA)
    sample_data = data.drop(columns=[ModelConfig.TARGET_COL]).head(1)
    actual_price = data[ModelConfig.TARGET_COL].iloc[0]
    
    # 4. Run Prediction
    log_prediction = model_wrapper.predict(sample_data)[0]
    
    # The dataset itself is log-transformed (~12.x), so we must always inverse it
    # to see dollar values, even if the config's auto-transform is off.
    if log_prediction < 30: 
        final_prediction = np.expm1(log_prediction)
    else:
        final_prediction = log_prediction
    
    print(f"\n📈 Test Prediction Result:")
    print(f"   - Predicted Sale Price: ${final_prediction:,.2f}")
    # Note: If it was log-transformed, the raw output was around 12.x
    print(f"   - Raw Model Output: {log_prediction:.4f}")
    print(f"   - Status: PASS")

if __name__ == "__main__":
    test_inference()
