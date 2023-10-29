# visualization.py

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from noise import apply_scanning_artifacts
import torch

def display_training_results(train_losses, valid_losses, model, data_loader, noise_params, device):
    # Display training and validation loss graph
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Fetch one batch of data
    images, _ = next(iter(data_loader))
    noisy_images = apply_scanning_artifacts(images, **noise_params)

    # Only take the first four images
    images = images[:4]
    noisy_images = noisy_images[:4]

    model.eval()
    with torch.no_grad():
        clean_images = model(noisy_images.to(device)).cpu()

    # Make a grid from images
    grid_original = vutils.make_grid(images, nrow=4, normalize=True, scale_each=True).cpu()
    grid_noisy = vutils.make_grid(noisy_images, nrow=4, normalize=True, scale_each=True).cpu()
    grid_cleaned = vutils.make_grid(clean_images, nrow=4, normalize=True, scale_each=True).cpu()

    # Display images
    plt.figure(figsize=(24, 12))  # Adjust figure size as needed
    plt.subplot(3, 1, 1)
    plt.title("Original Images")
    plt.imshow(grid_original.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.title("Noisy Images")
    plt.imshow(grid_noisy.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.title("Cleaned Images")
    plt.imshow(grid_cleaned.permute(1, 2, 0))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
