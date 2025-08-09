"""
Configuration file for wildfire project constants and hyperparameters.
"""

from pathlib import Path

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training parameters
BATCH_SIZE = 64
NUM_CLASSES = 1
EPOCHS = 10

# Model settings
MODEL_NAME = 'vgg16'  # Options: vgg16, resnet50, inceptionv3, efficientnetb3, proposed_model

# Directories (change these paths as per your system)
ROOT_DIR = Path("D:/Omkar/COEP BTech/BTech Sem VIII/Research AI/wildfire susceptibility")
DATASET_DIR = ROOT_DIR / "Dataset/Wildfirev2"
