#noise.py

import torch
import numpy as np
import cv2

def apply_affine_warp(image, intensity=0.5):

    h, w = image.shape[:2]
    
    # Define a maximum warp offset for the bottom-left point
    max_warp_offset = 10.0  # You can adjust this value based on the desired maximum warp

    # Calculate the actual warp offset
    warp_offset = intensity * max_warp_offset

    # Points in source image
    src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    
    # Corresponding points in output image - moving the bottom-left point according to intensity
    dst_points = np.float32([[0, 0], [w - 1, 0], [warp_offset, h - 1]])

    # Generating the affine matrix and applying it
    warp_matrix = cv2.getAffineTransform(src_points, dst_points)
    warped_image = cv2.warpAffine(image, warp_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return warped_image


def add_gaussian_noise(image, NOISE=0.4):

    mean = 0.0
    std = NOISE * np.amax(image)  # Max value in image determines noise strength
    noise = np.random.normal(mean, std, image.shape)
    
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clipping to maintain pixel value range
    return noisy_image

def add_scan_artifacts(image, SPECKLE=0.5, STREAK=0.5):

    h, w = image.shape[:2]

    # Define maximum number of speckles and streaks based on image size or as a constant
    max_speckle_count = int(0.01 * w * h)  # 1% of the number of pixels in the image
    max_streak_count = int(0.005 * w * h)  # 0.5% of the number of pixels

    # Calculate actual number of speckles and streaks to add
    speckle_count = int(SPECKLE * max_speckle_count)
    streak_count = int(STREAK * max_streak_count)

    # Add speckles (dust particles)
    for _ in range(speckle_count):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        image[y:y+2, x:x+2] = 255  # Vary the size and shape as needed

    # Add streaks
    for _ in range(streak_count):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        length, angle = np.random.randint(10, 20), np.random.uniform(0, 2*np.pi)  # Random length and angle
        x_end = int(x + length * np.cos(angle))
        y_end = int(y + length * np.sin(angle))
        image = cv2.line(image, (x, y), (x_end, y_end), (255, 255, 255), 1)  # White streaks

    return image


def apply_random_rotation(image, max_angle=30):
    # Generate a random angle using exponential distribution
    angle = np.random.exponential(scale=max_angle / 5)
    angle = min(angle, max_angle)  # Limiting the maximum angle

    # Deciding the rotation direction
    if np.random.rand() > 0.5:
        angle = -angle

    # Rotating the image
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return rotated_image

def apply_scanning_artifacts(data, noise=0.4, warp=0.5, speckle=0.2, streak=0.1, rotate=0.2):
    
    # Check if data is a tensor and convert to numpy if so
    was_tensor = False
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
        was_tensor = True
    
    # Assuming data is now in the form of a numpy array in BCHW format
    # Convert to HWC format for processing
    data_np = data.transpose(0, 2, 3, 1) * 255.0

    processed_images = []
    for img in data_np:
        img = img.astype(np.uint8)

        # Optionally apply random rotation
        if np.random.rand() < rotate:
            img = apply_random_rotation(img)

        # Apply warping
        warped_img = apply_affine_warp(img, intensity=warp)
        
        # Apply Gaussian noise
        noisy_img = add_gaussian_noise(warped_img, NOISE=noise)

        # Add dust and streaks
        artifact_img = add_scan_artifacts(noisy_img, SPECKLE=speckle, STREAK=streak)

        # Convert back to tensor format in CHW
        artifact_img = artifact_img.astype(np.float32) / 255.0
        artifact_img = artifact_img.transpose(2, 0, 1)  # Convert HWC back to CHW
        artifact_tensor = torch.tensor(artifact_img)
        processed_images.append(artifact_tensor)

    # Stack along the first dimension to create a batch
    output = torch.stack(processed_images)
    return output.to("mps")




