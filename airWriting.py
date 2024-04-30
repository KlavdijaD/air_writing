import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

NUM_EPOCHS = 10
nic = pd.read_csv('stevilka0.csv')
ena = pd.read_csv('stevilka1.csv')

#nic = nic.dropna()
#ena = ena.dropna()

#data = pd.concat([nic, ena])
ndArray = nic.to_numpy()
print(ndArray)

class AirWrite(nn.Module):
    def __init__(self):
        super(AirWrite, self).__init__()
        self.fc1 = nn.Linear(9, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

inputs = torch.tensor(ndArray[:, :-1], dtype=torch.float32)
labels = torch.tensor(ndArray[:, -1], dtype=torch.float32)

dataset = TensorDataset(inputs, labels)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

def train():
    model = AirWrite().to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        total = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)


            optimizer.zero_grad()
            outputs = model(inputs)
            print(f'Inputs Shape: {inputs.shape}')
            print(f'Labels Shape: {labels.shape}')
            print(f'Outputs Shape: {outputs.shape}')

            loss = loss_fn(outputs, labels)
            print("TUKAJ")
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    print(device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train()
    torch.save(model, "airWriting_model.pth")
