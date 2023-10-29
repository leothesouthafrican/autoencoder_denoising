# main.py

import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomImageDataset
from training_functions import train_denoising_model, validate_model
from autoencoders import Autoencoder1
import torch.optim as optim
from tqdm import tqdm
from noise import apply_scanning_artifacts
from visualisation import display_training_results

# Parse command line arguments for model continuation
parser = argparse.ArgumentParser(description="Train a denoising autoencoder.")
parser.add_argument('--model_path', type=str, default=None, 
                    help='Path to a saved model to continue training. Leave empty to train from scratch.')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate output directory name
output_dir = "output/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(output_dir, exist_ok=True)

# Set environment variables
TRAIN_FUNCTION = train_denoising_model
MODEL = Autoencoder1
EPOCHS = 150
DEVICE = "mps"

# Noise Parameters
NOISE_PARAMS = {
    "noise": 0.075,
    "warp": 0.9,
    "speckle": 0.7,
    "streak": 0.6,
    "rotate": 0.2
}

DATASET_PATH = "/Users/leo/Programming/autoencoder/data/TextImages/train_cleaned"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

full_dataset = CustomImageDataset(DATASET_PATH, transform=transform)

# Split dataset into train and validation sets
train_size = int(0.75 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Initialize or load model
if args.model_path and os.path.isfile(args.model_path):
    print(f"Loading model from {args.model_path}")
    model = MODEL().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
else:
    print("Training new model...")
    model = MODEL().to(DEVICE)

criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation
train_losses = []
valid_losses = []
min_valid_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), desc="Processing", total=len(train_loader), leave=True)
    for batch_idx, (data, _) in pbar:
        loss = TRAIN_FUNCTION(model, data, NOISE_PARAMS, optimizer, criterion, DEVICE)
        train_loss += loss
        pbar.set_postfix({'Batch Train Loss': f"{loss:.4f}"})

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    valid_loss = validate_model(model, valid_loader, NOISE_PARAMS, DEVICE)
    valid_losses.append(valid_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Train Loss: {avg_train_loss:.4f}, Average Validation Loss: {valid_loss:.4f}")

    # Check if this is the best model (lowest validation loss)
    if valid_loss < min_valid_loss:
        print(f"Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}). Saving model...")
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

# Display and save training results
display_training_results(train_losses, valid_losses, model, valid_loader, NOISE_PARAMS, DEVICE, output_dir)
