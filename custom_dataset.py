from PIL import Image
from torch.utils.data import Dataset
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB for consistency
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as dummy target, modify as needed
