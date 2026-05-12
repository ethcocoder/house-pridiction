# Local Interaction & Deployment Guide (Windows)

This guide provides instructions on how to interact with the House Price Prediction project locally on your Windows machine using PowerShell or Command Prompt.

---

## 1. Local Environment Setup

1. **Python Installation**: Ensure you have Python 3.9 or higher installed. 
   - Verify by running: `python --version`
2. **IDE Recommendation**: Use **VS Code** with the Python extension for the best experience.
3. **Navigate to Project**: Open your terminal in the `D:\house` directory.

## 2. Installation of Dependencies

Install all required libraries locally to ensure compatibility:

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

## 3. The Local Data Pipeline

If you change the raw data or need to refresh the processing logic, run the modular data script:

```powershell
# Clean data, engineer features, and save to data/processed/
python -m src.data.make_dataset
```
*Output logs can be found in `logs/data.log`.*

## 4. Training Models Locally
**Note:** Always retrain locally if you see "InconsistentVersionWarning" errors. This ensures the models are "baked" for your specific library versions.

```powershell
# 1. Train Baseline Model (Benchmark)
python -m src.models.baseline

# 2. Train XGBoost Model (Production - Includes Hyperparameter Tuning)
python -m src.models.xgboost_model

# 3. Train Deep Learning Model (PyTorch Neural Network)
python -m src.models.deep_learning
```

## 5. Running Inference Tests

Verify the accuracy of your trained models across multiple random scenarios:

```powershell
python -m src.models.test_model
```
This will output a table showing **Predicted Price**, **Actual Price**, and **Error %**.

## 6. Launching the Interactive Dashboard

Launch the premium Streamlit UI to interact with your models visually:

```powershell
streamlit run app/streamlit_app.py
```
*This will automatically open a new tab in your default browser at `http://localhost:8501`.*

## 7. Troubleshooting Version Mismatch

If you encounter an `AttributeError` or `InconsistentVersionWarning` when loading models trained in Colab:
1. Delete the old models in the `models/` folder.
2. Run the training commands in **Section 4** above.
3. The new models will be perfectly compatible with your local Scikit-Learn version.

---

## Quick Summary (Local Workflow)
1. `pip install -r requirements.txt`
2. `python -m src.data.make_dataset`
3. `python -m src.models.xgboost_model`
4. `streamlit run app/streamlit_app.py`
