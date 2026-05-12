# Google Colab Training & Deployment Instructions

This guide provides step-by-step instructions on how to train the Elite Residential Valuation Engine and deploy the Streamlit dashboard directly from Google Colab.

## 1. Setup the Colab Environment

1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. **Enable GPU (Recommended)**:
   - Click `Runtime` -> `Change runtime type`.
   - Select `T4 GPU` and click `Save`.

## 2. Clone the Repository

Instead of manual uploads, clone the project directly from GitHub into the Colab temporary storage:

```bash
# Clone the repository
!git clone https://github.com/ethcocoder/house-pridiction.git

# Navigate into the project folder
%cd house-pridiction
```

## 3. Install Dependencies

Install all required Python packages:

```bash
!pip install -r requirements.txt
```

## 4. Run the Data & Training Pipeline

Execute the modular scripts to process data and train the models:

```bash
# 1. Run Data Preprocessing & Feature Engineering
!python -m src.data.make_dataset

# 2. Train the XGBoost Model
!python -m src.models.xgboost_model

# 3. Train the Deep Learning Model
!python -m src.models.deep_learning
```

## 5. Deploy Streamlit App in Colab

You can run and view the Streamlit dashboard directly from Colab using `localtunnel`.

1. **Install Localtunnel**:
   ```bash
   !npm install -g localtunnel
   ```

2. **Run the App in the Background**:
   ```bash
   !streamlit run app/streamlit_app.py &>/dev/null &
   ```

3. **Expose the Port**:
   Run this cell to get your public URL:
   ```bash
   !npx localtunnel --port 8501
   ```
   *Click the link generated to open your dashboard. If prompted for an IP, run `!curl ipv4.icanhazip.com` to get the password.*

## 6. Save Results to Google Drive (Permanent Storage)

Colab files are deleted when the session ends. To save your trained models permanently:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create a folder in your Drive
!mkdir -p "/content/drive/MyDrive/House_Price_Project/models"

# Copy models and reports
!cp -r models/* "/content/drive/MyDrive/House_Price_Project/models/"
!cp -r reports/* "/content/drive/MyDrive/House_Price_Project/reports/"
```

## Summary Workflow
1. `!git clone https://github.com/ethcocoder/house-pridiction.git`
2. `!pip install -r requirements.txt`
3. `!python -m src.data.make_dataset`
4. `!python -m src.models.xgboost_model`
5. (Optional) Run Streamlit via `localtunnel`.
6. Sync `models/` to Google Drive.
