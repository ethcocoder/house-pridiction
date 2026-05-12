import pandas as pd
import numpy as np
import os
from src.models.xgboost_model import XGBoostModel
from src.config import PathConfig

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
    
    # 3. Create a dummy sample (matching the feature space)
    # Note: In a real scenario, you'd load a sample from data/processed/
    sample_data = pd.read_csv(PathConfig.PROCESSED_DATA).head(1).drop(columns=['SalePrice'])
    
    # 4. Run Prediction
    log_prediction = model_wrapper.predict(sample_data)
    final_prediction = np.expm1(log_prediction)[0]
    
    print(f"\n📈 Test Prediction Result:")
    print(f"   - Predicted Sale Price: ${final_prediction:,.2f}")
    print(f"   - Status: PASS")

if __name__ == "__main__":
    test_inference()
