#training_functions.py

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_denoising_model(model, data_loader, optimizer, criterion, device, apply_noise_func=None, **noise_params):
    model.train()
    total_loss = 0

    for batch in data_loader:
        data, _ = batch
        if apply_noise_func:
            # Convert tensor to numpy, apply noise, and convert back to tensor
            data_np = data.numpy().transpose(0, 2, 3, 1) * 255
            noisy_data_np = [apply_noise_func(img, **noise_params) for img in data_np]
            noisy_data = torch.tensor(np.array(noisy_data_np).transpose(0, 3, 1, 2) / 255, dtype=torch.float32)
            noisy_data = noisy_data.to(device)
        else:
            # Add default noise
            noisy_data = data + 0.1 * torch.randn_like(data, device=device)
            noisy_data = torch.clamp(noisy_data, 0., 1.)

        data = data.to(device)

        optimizer.zero_grad()
        output = model(noisy_data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate_model(model, valid_loader, device="cpu", apply_noise_func=None, **noise_params):
    model.eval()
    criterion = nn.MSELoss().to(device)
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(valid_loader):
            if apply_noise_func:
                # No need to convert to numpy and back if data is already a tensor
                noisy_data_np = [apply_noise_func(img.numpy().transpose(1, 2, 0) * 255, **noise_params) for img in data]
                noisy_data = torch.tensor(np.array(noisy_data_np).transpose(0, 3, 1, 2) / 255, dtype=torch.float32)
                noisy_data = noisy_data.to(device)
            else:
                noisy_data = data.to(device)

            data = data.to(device)

            output = model(noisy_data)
            loss = criterion(output, data)
            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    return avg_loss


