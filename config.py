"""
Configuration file for AttriDiffuser
"""

# Model settings
MODEL_CONFIG = {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "device": "auto",  # auto, cuda, or cpu
    "torch_dtype": "auto",  # auto, float16, or float32
}

# Generation settings
GENERATION_CONFIG = {
    "default_steps": 30,
    "default_guidance_scale": 7.5,
    "image_size": 512,
    "max_images": 4,
}

# Dataset settings
DATASET_CONFIG = {
    "default_path": r"C:\Users\hp\Downloads\drive-download-20260312T133839Z-3-003.zip",
    "extract_folder": "dataset",
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"],
}

# Attribute categories
ATTRIBUTES = {
    "gender": ["male", "female", "neutral"],
    "age": ["young", "middle-aged", "elderly", "child"],
    "expression": ["smiling", "serious", "neutral", "surprised", "sad"],
    "hair_color": ["black", "brown", "blonde", "red", "gray", "white"],
    "ethnicity": ["asian", "caucasian", "african", "hispanic", "middle-eastern"],
}

# UI settings
UI_CONFIG = {
    "page_title": "AttriDiffuser - Face Generation",
    "page_icon": "🎭",
    "layout": "wide",
}
