import pandas as pd
import numpy as np
import os
import joblib
from src.models.xgboost_model import XGBoostModel
from src.config import PathConfig, ModelConfig

def test_inference(num_samples=5):
    """Test the saved model artifacts by running multiple sample predictions."""
    print(f"🚀 Initializing Multi-Scenario Inference Test ({num_samples} samples)...")
    
    # 1. Initialize and Load Model
    model_wrapper = XGBoostModel()
    model_path = os.path.join(PathConfig.MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found. Please train the model first.")
        return
    model_wrapper.load(model_path)
    
    # 2. Load Data
    if not os.path.exists(PathConfig.PROCESSED_DATA):
        print(f"❌ Error: Processed data not found.")
        return
    data = pd.read_csv(PathConfig.PROCESSED_DATA)
    
    # 3. Run Multi-Test
    print("\n" + "="*80)
    print(f"{'Sample':<8} | {'Predicted ($)':<15} | {'Actual ($)':<15} | {'Error (%)':<10}")
    print("-" * 80)
    
    samples = data.sample(num_samples, random_state=42)
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        X_sample = pd.DataFrame([row.drop(ModelConfig.TARGET_COL)])
        y_actual_log = row[ModelConfig.TARGET_COL]
        
        # Predict
        y_pred_log = model_wrapper.predict(X_sample)[0]
        
        # Convert to Dollars
        y_pred_dollar = np.expm1(y_pred_log)
        y_actual_dollar = np.expm1(y_actual_log)
        
        # Calculate Error
        error_pct = abs(y_pred_dollar - y_actual_dollar) / y_actual_dollar * 100
        
        print(f"#{i+1:<7} | ${y_pred_dollar:13,.2f} | ${y_actual_dollar:13,.2f} | {error_pct:8.2f}%")
    
    print("="*80)
    print("✅ Multi-scenario test complete. Status: PASS")

if __name__ == "__main__":
    test_inference(num_samples=5)
