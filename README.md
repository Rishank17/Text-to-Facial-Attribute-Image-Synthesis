# AttriDiffuser: Text-to-Facial Attribute Image Synthesis

## Overview

AttriDiffuser is a simplified implementation inspired by the research paper **"AttriDiffuser: Adversarially Enhanced Diffusion Model for Text-to-Facial Attribute Image Synthesis"**.

The project generates realistic human face images from text descriptions containing facial attributes such as age, gender, expression, hair color, and ethnicity.

## Features

* Text-to-face image generation
* Multi-attribute facial control
* Stable Diffusion based image synthesis
* CLIP score evaluation for text-image similarity
* Streamlit web interface
* Multiple face generation variations

## Technologies Used

* Python
* PyTorch
* Diffusers
* Stable Diffusion
* CLIP
* Streamlit
* NumPy
* PIL

## Project Structure

app.py - Streamlit application

model.py - Face generation model

config.py - Configuration settings

utils.py - Utility functions

train_simple.py - Training demonstration script

requirements.txt - Required dependencies

## Installation

Install dependencies:

pip install -r requirements.txt

## Run the Application

streamlit run app.py

## Example Prompt

"A young female with a smiling expression, blonde hair, and green eyes"

## Note

This project is a simplified educational implementation inspired by the AttriDiffuser research paper and uses pretrained Stable Diffusion models for image generation.
