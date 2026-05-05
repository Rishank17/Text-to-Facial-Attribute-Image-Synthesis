"""
Utility functions for face image generation
Based on AttriDiffuser research paper
"""
import torch
import numpy as np
from PIL import Image
import os
from zipfile import ZipFile

def extract_dataset(zip_path, extract_to="dataset"):
    """
    Extract the dataset from zip file
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Dataset extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def load_images_from_folder(folder_path, max_images=100):
    """
    Load images from a folder
    
    Args:
        folder_path: Path to folder containing images
        max_images: Maximum number of images to load
    
    Returns:
        List of image paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= max_images:
                    return image_paths
    
    return image_paths

def preprocess_image(image_path, size=(256, 256)):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to image
        size: Target size (width, height)
    
    Returns:
        Preprocessed image tensor
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        img_array = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: Image tensor
    
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.clamp(0, 1)
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def parse_attributes(text_description):
    """
    Parse facial attributes from text description
    
    Args:
        text_description: Text describing facial attributes
    
    Returns:
        Dictionary of parsed attributes
    """
    attributes = {
        'gender': None,
        'age': None,
        'expression': None,
        'hair_color': None,
        'ethnicity': None
    }
    
    text_lower = text_description.lower()
    
    # Gender
    if 'male' in text_lower and 'female' not in text_lower:
        attributes['gender'] = 'male'
    elif 'female' in text_lower:
        attributes['gender'] = 'female'
    
    # Age
    if 'young' in text_lower or 'teen' in text_lower:
        attributes['age'] = 'young'
    elif 'old' in text_lower or 'elderly' in text_lower:
        attributes['age'] = 'old'
    elif 'middle' in text_lower:
        attributes['age'] = 'middle-aged'
    
    # Expression
    if 'smiling' in text_lower or 'smile' in text_lower:
        attributes['expression'] = 'smiling'
    elif 'serious' in text_lower:
        attributes['expression'] = 'serious'
    
    return attributes
