# Google Colab Training Instructions

This guide provides step-by-step instructions on how to train the Elite Residential Valuation Engine (XGBoost & Deep Learning models) using Google Colab's cloud computing resources.

## 1. Prepare the Project for Colab

Since you are moving from a local environment to Colab, the easiest way is to upload the project directly or use Google Drive.

1. **Zip the Project**: Compress the entire `project-repository` (the `house` folder) into a `.zip` file. Do not include the `data/raw` files if they are too large; instead, upload them separately if needed, or include them if they are small enough.
2. **Upload to Google Drive**: 
   - Open your Google Drive.
   - Create a folder named `Colab Projects`.
   - Upload your `house.zip` file into this folder.

## 2. Setup the Colab Environment

1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. **Enable GPU (Optional but Recommended for Deep Learning)**:
   - Click `Runtime` -> `Change runtime type` in the top menu.
   - Select `T4 GPU` (or similar) from the Hardware accelerator dropdown.
   - Click `Save`.

## 3. Mount Google Drive and Extract Files

In the first cell of your new Colab notebook, mount your Google Drive so Colab can access your uploaded zip file:

```python
from google.colab import drive
drive.mount('/content/drive')
```
*(You will be prompted to authorize access to your Google Drive).*

In the next cell, unzip the project into the Colab environment:

```bash
# Unzip the project (adjust the path if you named the zip or folder differently)
!unzip "/content/drive/MyDrive/Colab Projects/house.zip" -d "/content/house"
```

## 4. Install Dependencies

Navigate into the project directory and install the required Python packages from the `requirements.txt` file:

```bash
# Change directory to the project root
%cd /content/house

# Install dependencies
!pip install -r requirements.txt
```

## 5. Run the Data Pipeline

Before training the models, you must run the data preprocessing and feature engineering pipeline to generate the cleaned dataset.

```bash
# Ensure you are still in the /content/house directory
!python -m src.data.make_dataset
```
*This will read from `data/raw/housing.csv` and create `data/processed/housing_cleaned.csv`.*

## 6. Train the Models

Now you can train the models. 

**To train the Advanced XGBoost Model:**
```bash
!python -m src.models.xgboost_model
```

**To train the Deep Learning (MLP) Model (if applicable):**
```bash
!python -m src.models.deep_learning
```

## 7. Evaluate and Save Results

You can run the evaluation notebook directly in Colab, or run the baseline model to compare:

```bash
!python -m src.models.baseline
```

### Saving Artifacts Back to Google Drive
Colab environments are ephemeral (they delete data when closed). You **must** copy your trained models and processed data back to your Google Drive to save them permanently.

Run this cell at the end of your training:

```bash
# Copy trained models back to Drive
!cp -r /content/house/models/* "/content/drive/MyDrive/Colab Projects/saved_models/"

# Copy any generated reports/figures
!cp -r /content/house/reports/* "/content/drive/MyDrive/Colab Projects/saved_reports/"
```

## Summary Workflow
1. Zip local project -> Upload to Drive.
2. Mount Drive in Colab -> Unzip.
3. `%cd /content/house` -> `!pip install -r requirements.txt`.
4. `!python -m src.data.make_dataset`.
5. `!python -m src.models.xgboost_model`.
6. `!cp -r models/* /content/drive/MyDrive/Colab Projects/`.
