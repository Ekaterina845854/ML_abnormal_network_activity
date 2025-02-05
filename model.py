import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
from capture import extract_features
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Определение автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),       
            nn.ELU(),
            nn.BatchNorm1d(256),            
            nn.Dropout(0.2),                 
            nn.Linear(256, 128),             
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),              
            nn.ELU(),
            nn.BatchNorm1d(64)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Функция для вычисления ошибки восстановления
def compute_reconstruction_error(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Функция для тренировки модели
def train_autoencoder(train_dataset, test_dataset, input_dim, num_epochs=50, batch_size=128, learning_rate=1e-3):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Autoencoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for data in train_loader:
            inputs = data[0]
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    reconstruction_error = compute_reconstruction_error(model, test_loader)
    print(f"Reconstruction error on test set: {reconstruction_error:.4f}")

    return model


# Функция для обработки пользовательского датасета
def process_and_train(dataset_path):
    # Чтение и обработка датасета
    df = pd.read_csv(dataset_path)
    feature_matrix = extract_features(df)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)

    X_train, X_test = train_test_split(features_scaled, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    input_dim = X_train.shape[1]

    # Тренировка автоэнкодера
    model = train_autoencoder(train_dataset, test_dataset, input_dim, num_epochs=500, batch_size=512, learning_rate=1e-3)

    # Сохранение модели
    model_path = "autoencoder_model_111ined.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    # Вычисление ошибок восстановления для тестового набора
    reconstruction_errors = []
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0]
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, inputs)
            reconstruction_errors.append(loss.item())

    reconstruction_errors = np.array(reconstruction_errors)

    # Вычисление порога аномалий на основе Z-оценки
    mean = np.mean(reconstruction_errors)
    std_dev = np.std(reconstruction_errors)

    z = 3
    threshold = mean + z * std_dev

    print(f"Calculated threshold: {threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder Model on a custom dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset CSV file")
    args = parser.parse_args()

    process_and_train(args.dataset)
