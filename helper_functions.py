import matplotlib.pyplot as plt
import torch
from autoencoders import Autoencoder1, Autoencoder2, Autoencoder3

def display_denoising_images(n, images, model, device, apply_noise_func, **noise_params):
    # Calculate the dimensions of the input images dynamically
    shape = images.shape[1:]  # For example, shape will be [3, 32, 32] for CIFAR-100

    # Apply the custom noise function to the images
    noisy_images = apply_noise_func(images, **noise_params)
    noisy_images = torch.clamp(noisy_images, 0., 1.)

    # Handle the two models differently
    if isinstance(model, Autoencoder1):
        # Flatten the image if the model is Autoencoder1
        noisy_images_flatten = noisy_images.view(noisy_images.size(0), -1).to(device)
        with torch.no_grad():
            decoded = model(noisy_images_flatten).view(n, *shape)
    elif isinstance(model, Autoencoder2) or isinstance(model, Autoencoder3):
        # Keep the shape as is for Autoencoder2
        noisy_images = noisy_images.to(device)
        with torch.no_grad():
            decoded = model(noisy_images)

    # Function to subplot an image
    def subplot_image(ax, img, title):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())  # Normalize the image for display
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    # Create a new figure
    plt.figure(figsize=(18, 6))

    # Display original, noisy, and denoised images
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        subplot_image(ax, images[i], 'Original')

        ax = plt.subplot(3, n, i + 1 + n)
        subplot_image(ax, noisy_images[i], 'Noisy')

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        subplot_image(ax, decoded[i], 'Denoised')

    plt.tight_layout()
    plt.show()

