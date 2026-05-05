"""
Simplified AttriDiffuser Model Implementation
Based on the research paper: AttriDiffuser - Text-to-Facial Attribute Image Synthesis
"""
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

class SimplifiedAttriDiffuser:
    """
    Simplified version of AttriDiffuser for text-to-face generation
    Uses pre-trained Stable Diffusion as base model
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the model
        
        Args:
            device: Device to run model on (cuda or cpu)
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Realistic Vision V5.1 - fine-tuned specifically on realistic human faces
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        vae_id = "stabilityai/sd-vae-ft-mse"  # sharper facial details

        try:
            print("Loading Realistic Vision V5.1 model (face-specialized)...")

            from diffusers import AutoencoderKL
            from diffusers import DPMSolverMultistepScheduler

            # Load face-specific VAE for sharper details
            vae = AutoencoderKL.from_pretrained(
                vae_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )

            # Use DPMSolver scheduler - same quality at fewer steps (faster + better)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                final_sigmas_type="sigma_min"
            )

            self.pipe = self.pipe.to(device)

            # Load face enhancer LoRA for better facial detail
            print("Loading face enhancer LoRA...")
            try:
                self.pipe.load_lora_weights(
                    "imagepipeline/better-face-enhancer-v3",
                    weight_name="better-face-enhancer-v3.safetensors"
                )
                self.pipe.fuse_lora(lora_scale=0.7)  # 0.7 = strong but not overdone
                print("Face enhancer LoRA loaded!")
            except Exception as lora_err:
                print(f"LoRA load skipped: {lora_err}")

            # Enable memory optimizations for CPU
            if device == "cpu":
                print("Enabling CPU optimizations...")
                self.pipe.enable_attention_slicing(1)
                if hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.device = "cpu"
            self.pipe = None
    
    def generate_face(self, text_prompt, num_inference_steps=30, guidance_scale=7.5, callback=None, fast_mode=False):
        """
        Generate face image from text description
        
        Args:
            text_prompt: Text description of facial attributes
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            callback: Optional callback function(step, total_steps) for progress tracking
            fast_mode: If True, generates 256x256 image for faster processing
        
        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            print("Model not loaded. Please check your installation.")
            return None
        
        try:
            # Enhanced prompt for Realistic Vision - face-specialized model
            enhanced_prompt = (
                f"RAW photo, a close up portrait of a face, {text_prompt}, "
                f"(high detailed skin:1.2), 8k uhd, dslr, soft lighting, "
                f"high quality, film grain, Fujifilm XT3"
            )

            # Strong negative prompt to avoid common face generation artifacts
            negative_prompt = (
                "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
                "sketch, cartoon, drawing, anime, mutated hands and fingers, "
                "deformed, distorted, disfigured, poorly drawn, bad anatomy, "
                "wrong anatomy, extra limb, missing limb, floating limbs, "
                "disconnected limbs, mutation, mutated, ugly, disgusting, "
                "blurry, amputation, watermark, signature, text"
            )
            
            # Define callback wrapper for progress
            def progress_callback(step, timestep, latents):
                if callback:
                    callback(step + 1, num_inference_steps)
            
            # Set image dimensions based on mode
            img_size = 256 if fast_mode else 512
            
            # Generate image
            with torch.no_grad():
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=img_size,
                    width=img_size,
                    callback=progress_callback,
                    callback_steps=1
                )
            
            return result.images[0]
        
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def generate_multiple_faces(self, text_prompt, num_images=4, num_inference_steps=30):
        """
        Generate multiple diverse faces from same description
        Implements the diversity aspect from AttriDiffuser
        
        Args:
            text_prompt: Text description
            num_images: Number of images to generate
            num_inference_steps: Denoising steps
        
        Returns:
            List of generated images
        """
        images = []
        
        for i in range(num_images):
            print(f"Generating image {i+1}/{num_images}...")
            # Vary guidance scale for diversity
            guidance = 7.5 + (i * 0.5)
            img = self.generate_face(text_prompt, num_inference_steps, guidance)
            if img:
                images.append(img)
        
        return images


class AttributeEncoder(nn.Module):
    """
    Simple attribute encoder for multi-attribute control
    Based on AttriDiffuser's attribute gating mechanism
    """
    
    def __init__(self, num_attributes=5, embedding_dim=512):
        super().__init__()
        self.num_attributes = num_attributes
        self.embedding_dim = embedding_dim
        
        # Attribute embeddings
        self.attribute_embeddings = nn.ModuleDict({
            'gender': nn.Embedding(3, embedding_dim),  # male, female, neutral
            'age': nn.Embedding(4, embedding_dim),     # young, middle, old, unknown
            'expression': nn.Embedding(5, embedding_dim),  # smile, serious, neutral, etc.
            'hair': nn.Embedding(6, embedding_dim),    # various hair colors
            'ethnicity': nn.Embedding(5, embedding_dim)  # various ethnicities
        })
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
    
    def forward(self, attributes):
        """
        Encode attributes with gating
        
        Args:
            attributes: Dictionary of attribute indices
        
        Returns:
            Gated attribute embeddings
        """
        embeddings = []
        
        for attr_name, attr_value in attributes.items():
            if attr_name in self.attribute_embeddings and attr_value is not None:
                emb = self.attribute_embeddings[attr_name](attr_value)
                gated_emb = emb * self.gate(emb)
                embeddings.append(gated_emb)
        
        if embeddings:
            return torch.stack(embeddings).mean(dim=0)
        else:
            return torch.zeros(self.embedding_dim)
