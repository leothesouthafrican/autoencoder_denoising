from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train_denoising_model(model, data, noise_factor, optimizer, criterion, device, should_flatten=True):
    # Add noise to the data
    noisy_data = data + noise_factor * torch.randn(data.shape)
    noisy_data = torch.clamp(noisy_data, 0., 1.)

    if should_flatten:
        # Flatten the data for Autoencoder1
        noisy_data = noisy_data.view(noisy_data.size(0), -1).to(device)
        data = data.view(data.size(0), -1).to(device)
    else:
        # Keep data as is for Autoencoder2
        noisy_data = noisy_data.to(device)
        data = data.to(device)

    optimizer.zero_grad()
    output = model(noisy_data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()

    return loss


def train_model(model, data, optimizer, criterion, device, should_flatten=True):
    if should_flatten:
        data = data.view(data.size(0), -1).to(device)

    optimizer.zero_grad()
    output = model(data.to(device))
    loss = criterion(output, data.to(device))
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_model(model, valid_loader, noise_factor=0.0, device="cpu", should_flatten=True):
    model.to(device)
    model.eval()  # set model to evaluation mode
    criterion = nn.MSELoss().to(device)
    total_loss = 0

    with torch.no_grad():  # we don't need gradients for validation
        for batch_idx, (data, _) in enumerate(valid_loader):
            noisy_data = data + noise_factor * torch.randn(*data.shape)
            noisy_data = torch.clamp(noisy_data, 0., 1.)
            
            if should_flatten:
                data = data.view(data.size(0), -1).to(device)
                noisy_data = noisy_data.view(noisy_data.size(0), -1).to(device)

            output = model(noisy_data.to(device))
            loss = criterion(output, data.to(device))
            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    model.train()  # set model back to training mode
    return avg_loss
