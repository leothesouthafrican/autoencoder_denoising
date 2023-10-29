from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train_denoising_model(model, data, noise_factor, optimizer, criterion, device):
    # Add noise to the data
    noisy_data = data + noise_factor * torch.randn_like(data)
    noisy_data = torch.clamp(noisy_data, 0., 1.).to(device)
    data = data.to(device)

    optimizer.zero_grad()
    output = model(noisy_data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()

    return loss

def validate_model(model, valid_loader, noise_factor=0.0, device="cpu"):
    model.to(device)
    model.eval()  # set model to evaluation mode
    criterion = nn.MSELoss().to(device)
    total_loss = 0

    with torch.no_grad():  # no gradients needed for validation
        for data, _ in valid_loader:
            noisy_data = data + noise_factor * torch.randn_like(data)
            noisy_data = torch.clamp(noisy_data, 0., 1.).to(device)
            data = data.to(device)

            output = model(noisy_data)
            loss = criterion(output, data)
            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    model.train()  # set model back to training mode
    return avg_loss