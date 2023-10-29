# training_functions.py
import torch
import torch.nn as nn
import torch.optim as optim
from noise import apply_scanning_artifacts

def train_denoising_model(model, data, noise_params, optimizer, criterion, device):
    # Add custom noise to the data
    noisy_data = apply_scanning_artifacts(data, **noise_params).to(device)
    data = data.to(device)

    optimizer.zero_grad()
    output = model(noisy_data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_model(model, valid_loader, noise_params, device="cpu"):
    model.to(device)
    model.eval()
    criterion = nn.MSELoss().to(device)
    total_loss = 0

    with torch.no_grad():
        for data, _ in valid_loader:
            noisy_data = apply_scanning_artifacts(data, **noise_params).to(device)
            data = data.to(device)

            output = model(noisy_data)
            loss = criterion(output, data)
            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    model.train()
    return avg_loss
