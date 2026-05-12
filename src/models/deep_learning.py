import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib
from src.core.base import ModelInterface
from src.data.utils import setup_logger
from src.config import PathConfig, ModelConfig

logger = setup_logger("DeepLearningModel", log_file=os.path.join(PathConfig.LOGS_DIR, "modeling.log") if os.path.exists(PathConfig.LOGS_DIR) else "modeling.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HousingDataset(Dataset):
    """Custom Dataset for loading tabular housing data into PyTorch."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HousingMLP(nn.Module):
    """Multi-Layer Perceptron architecture for regression."""
    def __init__(self, input_size):
        super(HousingMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class DeepLearningModel(ModelInterface):
    """Professional Deep Learning wrapper implementing the ModelInterface."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.input_size = None
        self.best_state = None

    def _prepare_data(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
        return self.preprocessor.fit_transform(X)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Preparing data for Deep Learning training...")
        X_processed = self._prepare_data(X)
        self.input_size = X_processed.shape[1]
        
        X_train, X_val, y_train, y_val = train_test_split(X_processed, y.values, test_size=0.2, random_state=ModelConfig.RANDOM_STATE)
        
        train_loader = DataLoader(HousingDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(HousingDataset(X_val, y_val), batch_size=64, shuffle=False)
        
        self.model = HousingMLP(self.input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        logger.info(f"Starting Neural Network training on {device}...")
        epochs = 100
        best_loss = float('inf')
        
        pbar = tqdm(range(epochs), desc="Training Epochs")
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_loss += criterion(self.model(batch_X), batch_y).item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.best_state = self.model.state_dict()
            
            pbar.set_postfix({'Val_Loss': f'{avg_val_loss:.4f}', 'Best': f'{best_loss:.4f}'})
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}")

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        logger.info("Training complete. Best weights loaded.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model is not trained or loaded yet.")
        
        self.model.eval()
        X_processed = torch.tensor(self.preprocessor.transform(X), dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model(X_processed).cpu().numpy().flatten()
        return preds

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model state and preprocessor
        save_dict = {
            'model_state': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'input_size': self.input_size
        }
        torch.save(save_dict, path)
        logger.info(f"Deep Learning model artifacts saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=device)
        self.input_size = checkpoint['input_size']
        self.preprocessor = checkpoint['preprocessor']
        self.model = HousingMLP(self.input_size).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Deep Learning model loaded from {path}")

if __name__ == "__main__":
    # 1. Load Processed Data
    if not os.path.exists(PathConfig.PROCESSED_DATA):
        print(f"Error: Processed data not found at {PathConfig.PROCESSED_DATA}. Run 'python -m src.data.make_dataset' first.")
    else:
        df = pd.read_csv(PathConfig.PROCESSED_DATA)
        X = df.drop(columns=[ModelConfig.TARGET_COL])
        if ModelConfig.LOG_TRANSFORM:
            y = np.log1p(df[ModelConfig.TARGET_COL])
        else:
            y = df[ModelConfig.TARGET_COL]
        
        # 2. Train and Save
        model = DeepLearningModel()
        model.train(X, y)
        model.save(os.path.join(PathConfig.MODELS_DIR, "deep_learning_model.pth"))
