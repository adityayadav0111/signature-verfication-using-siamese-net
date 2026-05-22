import os
import torch
from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(100, 100)):
    """
    Preprocess the image to match the training pipeline:
    1. Resize to (100, 100)
    2. Convert to grayscale
    3. Normalize to [0, 1]
    4. Convert to tensor
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to (100, 100)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.from_numpy(img).float()  # Convert to tensor
    img = img.unsqueeze(0)  # Add channel dimension
    return img

def compare_signatures(model, image1, image2):
    """
    Compare two signatures using the Siamese model.
    Returns the Euclidean distance between the two signatures.
    """
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    return euclidean_distance.item()